# Developing BentoML Applications on Kubeflow

Starting with the release of Kubeflow 1.7, BentoML provides a native integration with Kubeflow. This integration enables the seamless packaging and deployment of models trained in the Kubeflow notebook or pipeline as [Bentos](https://docs.bentoml.org/en/latest/concepts/bento.html). These services can then be containerized into OCI images and deployed to a Kubernetes cluster in a microservice architecture through BentoML's cloud native components and custom resource definitions (CRDs). This documentation provides a comprehensive guide on how to use BentoML and Kubeflow together to streamline the process of deploying machine learning models at scale.

In this example, we will train three fraud detection models using the [Kaggle IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) using the Kubeflow notebook and create a BentoML service that simultaneously invoke all three models and returns the decision if any one of the models predicts that a transactin is a fraud. We will build and push the BentoML service to an S3 bucket. Next we will containerize BentoML service from the S3 bucket and deploy the service to Kubeflow cluster using using BentoML custom resource definitions on Kubernetes. The service will be deployed in a microservice architecture with each model running in a separate pod, deployed on hardware that is the most ideal for running the model, and scale independently.

This example can also be run from the `notebook.ipynb` included in this directory.

## Prerequisites

This guide assume that Kubeflow is already installed in Kubernetes cluster. See [Kubeflow Manifests](https://github.com/kubeflow/manifests) for installation instructions.

Install BentoML cloud native components and custom resource definitions.

```bash
kustomize build bentoml-yatai-stack/default | kubectl apply -n kubeflow --server-side -f -
```

Install the required packages to run this example.

```bash
pip install -r requirements.txt
```

## Download Kaggle Dataset

Set Kaggle [user credentials](https://github.com/Kaggle/kaggle-api#api-credentials) for API access. Accepting the [rules of the competition](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) is required for downloading the dataset.

```bash
export KAGGLE_USERNAME=
export KAGGLE_KEY=
```

Download Kaggle dataset.

```bash
kaggle competitions download -c ieee-fraud-detection
rm -rf ./data/
unzip -d ./data/ ieee-fraud-detection.zip && rm ieee-fraud-detection.zip
```

## Train Models

In this demonstration, we'll train three fraud detection models using the Kaggle IEEE-CIS Fraud Detection dataset. To showcase saving and serving multiple models with Kubeflow and BentoML, we'll split the dataset into three equal-sized chunks and use each chunk to train a separate model. While this approach has no practical benefits, it will help illustrate how to save and serve multiple models with Kubeflow and BentoML.

```python
import pandas as pd

df_transactions = pd.read_csv("./data/train_transaction.csv")

X = df_transactions.drop(columns=["isFraud"])
y = df_transactions.isFraud
```

Define the preprocessor.

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

numeric_features = df_transactions.select_dtypes(include="float64").columns
categorical_features = df_transactions.select_dtypes(include="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",
)

X = preprocessor.fit_transform(X)
```

Define our training function with the number of boosting rounds and maximum depths.

```python
import xgboost as xgb

def train(n_estimators, max_depth):
    return xgb.XGBClassifier(
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        eval_metric="aucpr",
        objective="binary:logistic",
        enable_categorical=True,
    ).fit(X_train, y_train, eval_set=[(X_test, y_test)])
```

We will divide the training data into three equal-sized chunks and treat them as independent data sets. Based on these data sets, we will train three separate fraud detection models. The trained model will be saved to the local model store using BentoML model saving API.

```python
import bentoml

from sklearn.model_selection import train_test_split

CHUNKS = 3
CHUNK_SIZE = len(X) // CHUNKS

for i in range(CHUNKS):
    START = i * CHUNK_SIZE
    END = (i + 1) * CHUNK_SIZE
    X_train, X_test, y_train, y_test = train_test_split(X[START:END], y[START:END])

    name = f"ieee-fraud-detection-{i}"
    model = train(10, 5)
    score = model.score(X_test, y_test)
    print(f"Successfully trained model {name} with score {score}.")

    bentoml.xgboost.save_model(
        name,
        model,
        signatures={
            "predict_proba": {"batchable": True},
        },
        custom_objects={"preprocessor": preprocessor},
    )
    print(f"Successfully saved model {name} to the local model store.")
```

Saved models can be loaded back into the memory and debugged in the notebook.

```python
import bentoml
import pandas as pd
import numpy as np

model_ref = bentoml.xgboost.get("ieee-fraud-detection-0:latest")
model_runner = model_ref.to_runner()
model_runner.init_local()
model_preprocessor = model_ref.custom_objects["preprocessor"]

test_transactions = pd.read_csv("./data/test_transaction.csv")[0:500]
test_transactions = model_preprocessor.transform(test_transactions)
result = model_runner.predict_proba.run(test_transactions)
np.argmax(result, axis=1)
```

## Define Service API

After the models are built and scored, let's create the service definition. You can find the service definition in the `service.py` module in this example. Let's breakdown the `service.py` module and explain what each section does.

First, we will create a list of preprocessors and runners from the three models we saved earlier. Runners are abstractions of the model inferences that can be scaled independently. See [Using Runners](https://docs.bentoml.org/en/latest/concepts/runner.html) for more details.

```python
fraud_detection_preprocessors = []
fraud_detection_runners = []

for model_name in ["ieee-fraud-detection-0", "ieee-fraud-detection-1", "ieee-fraud-detection-2"]:
    model_ref = bentoml.xgboost.get(model_name)
    fraud_detection_preprocessors.append(model_ref.custom_objects["preprocessor"])
    fraud_detection_runners.append(model_ref.to_runner())
```

Next, we will create a service with the list of runners passed in.

```python
svc = bentoml.Service("fraud_detection", runners=fraud_detection_runners)
```

Finally, we will create the API function `is_fraud`. We'll use the `@api` decorator to declare that the function is an API and specify the input and output types as pandas.DataFrame and JSON, respectively. The function is defined as `async` so that the inference calls to the runners can happen simultaneously without waiting for the results to return before calling the next runner. The inner function `_is_fraud` defines the model inference logic for each runner. All runners are called simultaneously through the `asyncio.gather` function and the results are aggregated into a list. The function will return True if any of the models return True.

For more about service definitinos, please see [Service and APIs](https://docs.bentoml.org/en/latest/concepts/service.html).

## Build Service

Building the service and models into a bento allows it to be distributed among collaborators, containerized into a OCI image, and deployed in the Kubernetes cluster. To build a service into a bento, we first need to define the `bentofile.yaml` file. See [Building Bentos](https://docs.bentoml.org/en/latest/concepts/bento.html) for more options.

```yaml
service: "service:svc"
include:
- "service.py"
- "sample.py"
python:
  requirements_txt: ./requirements.txt
```

Running the following command will build the service into a bento and store it to the local bento store.

```bash
bentoml build
```

## Serve Bento

Serving the bento will bring up a service endpoint in HTTP or gRPC for the service API we defined. Use `--help` to see more serving options.

```bash
bentoml serve-http --production
```

## Containerize and Push Image

Containerize the image through `docker build` and `push` the image to a remote repository of your choice.

```bash
bentoml containerize fraud_detection:zsk3powbr2adgcvj -t your-username/fraud_detection:zsk3powbr2adgcvj
```

```bash
docker push your-username/fraud_detection:zsk3powbr2adgcvj
```

## Deploy to Kubernetes Cluster

BentoML offers three custom resource definitions (CRDs) in the Kubernetes cluster.

- [BentoRequest](https://docs.bentoml.org/projects/yatai/en/latest/concepts/bentorequest_crd.html)
- [Bento](https://docs.bentoml.org/projects/yatai/en/latest/concepts/bento_crd.html)
- [BentoDeployment](https://docs.bentoml.org/projects/yatai/en/latest/concepts/bentodeployment_crd.html)

Since we have already built and pushed image in the earlier step, we will only need to create the `Bento` and `BentoDeployment` custom resources. The `Bento` CRD describes the metadata for the Bento such as the address of the image and the runners. The `BentoDeployment` CRD describes the metadata of the deployment such as resources and autoscaling behaviors. `BentoDeployment` requires `Bento` to be ready and will reconcile a `Bento` CR of the same name before a deployment is created. See `deployment.yaml` file included for an example.

The `BentoRequest` CRD describes the meta data needed for building the image such as the download URL of the Bento. Creating the `BentoRequest` CR is not needed in this example.

Apply the `Bento` and `BentoDeployment` CRDs.

```bash
kubectl apply -f deployment.yaml
```

Verify the `Bento` and `BentoDeployment` resources. Note that API server and runners are run in separate pods and created in separate deployments that can be scaled independently.

```bash
kubectl -n kubeflow get pods -l yatai.ai/bento-deployment=fraud-detection
```

Port forward the Fraud Detection service to test locally. You should be able to visit the Swagger page of the service by requesting http://0.0.0.0:8080 while port forwarding.

```bash
kubectl -n kubeflow port-forward svc/fraud-detection 8080:3000 --address 0.0.0.0
```

Delete the `Bento` and `BentoDeployment` resources.

```bash
kubectl delete -f deployment.yaml
```

# Conclusion

Congratulations! You completed the example. Let's recap what we have learned.

- Trained three fraud detection models and saved them to the BentoML model store.
- Created a BentoML service that runs inferences on all three models simultaneously, combines and returns the results.
- Containerized the BentoML service into an OCI image and pushed the image to a remote repository.
- Created BentoML CRDs on Kubernetes to deploy the Bento in a microservice architecture.
