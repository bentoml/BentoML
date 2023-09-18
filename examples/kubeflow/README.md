# Developing BentoML Applications on Kubeflow

Starting with the release of Kubeflow 1.7, BentoML provides a native integration with Kubeflow. This integration allows you to package models trained in Kubeflow Notebooks or Pipelines as [Bentos](https://docs.bentoml.org/en/latest/concepts/bento.html), and deploy them as microservices in the Kubernetes cluster through BentoML's cloud native components and custom resource definitions (CRDs). This documentation provides a comprehensive guide on how to use BentoML and Kubeflow together to streamline the process of deploying models at scale.

In this example, we will train three fraud detection models using the Kubeflow notebook and the [Kaggle IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). We will then create a BentoML service that can simultaneously invoke all three models and return a decision on whether a transaction is fraudulent and build it into a Bento. We will showcase two deployment workflows using BentoML's Kubernetes operators: deploying directly from the Bento, and deploying from an OCI image built from the Bento.

## Prerequisites

Install Kubeflow and BentoML resources to the Kubernetes cluster. See [Kubeflow](https://github.com/kubeflow/manifests) and [BentoML](https://github.com/kubeflow/manifests/tree/master/contrib/bentoml) manifests installation guides for details.

After BentoML Kubernetes resources are installed successfully, you should have the following CRDs in the namespace.

```bash
> kubectl -n kubeflow get crds | grep bento
bentodeployments.serving.yatai.ai                     2022-12-22T18:46:46Z
bentoes.resources.yatai.ai                            2022-12-22T18:46:47Z
bentorequests.resources.yatai.ai                      2022-12-22T18:46:47Z
```

Install the required packages to run this example.

```bash
git clone --depth 1 https://github.com/bentoml/BentoML
cd BentoML/examples/kubeflow
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

In this demonstration, we'll train three fraud detection models using the Kaggle IEEE-CIS Fraud Detection dataset. To showcase saving and serving multiple models with Kubeflow and BentoML, we'll split the dataset into three equal-sized chunks and use each chunk to train a separate model. While this approach has no practical benefits, it will help illustrate how to save and serve multiple models with Kubeflow and BentoML. This step can also be run from the `notebook.ipynb` included in this directory.

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

```python
@svc.api(input=PandasDataFrame.from_sample(sample_input), outp1t=JSON())
async def is_fraud(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)

    async def _is_fraud(preprocessor, runner, input_df):
        input_features = preprocessor.transform(input_df)
        results = await runner.predict_proba.async_run(input_features)
        predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
        return bool(predictions[0])

    # Simultaeously run all models
    results = await asyncio.gather(
        *[
            _is_fraud(p, r, input_df)
            for p, r in zip(fraud_detection_preprocessors, fraud_detection_runners)
        ]
    )

    # Return fraud if at least one model returns fraud
    return any(results)
```

For more about service definitions, please see [Service and APIs](https://docs.bentoml.org/en/latest/concepts/service.html).

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
bentoml serve-http
```

## Deploy to Kubernetes Cluster

BentoML offers three custom resource definitions (CRDs) in the Kubernetes cluster.

- [BentoRequest](https://docs.bentoml.org/projects/yatai/en/latest/concepts/bentorequest_crd.html) - Describes the metadata needed for building the container image of the Bento, such as the download URL. Created by the user.
- [Bento](https://docs.bentoml.org/projects/yatai/en/latest/concepts/bento_crd.html) - Describes the metadata for the Bento such as the address of the image and the runners. Created by users or by the `yatai-image-builder` operator for reconsiliating `BentoRequest` resources.
- [BentoDeployment](https://docs.bentoml.org/projects/yatai/en/latest/concepts/bentodeployment_crd.html) - Describes the metadata of the deployment such as resources and autoscaling behaviors. Reconciled by the `yatai-deployment` operator to create Kubernetes deployments of API Servers and Runners.

![image](https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/img/kubeflow-crds.png)

Next, we will demonstrate two ways of deployment.

1. Deploying using a `BentoRequest` resource by providing a Bento
2. Deploying Using a `Bento` resource by providing a pre-built container image from a Bento

### Deploy with BentoRequest CRD

In this workflow, we will export the Bento to a remote storage, and then use the `yatai-image-builder` operator to containerize it. Finally, we'll deploy the containerized Bento image using the `yatai-deployment` operator. Using the `yatai-image-builder` operator to download Bentos from AWS and push the containerized OCI image will require setting up the credentials of S3 bucket and container repository. See the [manifests installation guide](https://github.com/kubeflow/manifests/tree/master/contrib/bentoml) for detailed instructions.

Push the Bento built and saved in the local Bento store to a cloud storage such as AWS S3.

```bash
bentoml export fraud_detection:o5smnagbncigycvj s3://your_bucket/fraud_detection.bento
```

Apply the `BentoRequest` and `BentoDeployment` resources as defined in `deployment_from_bentorequest.yaml` included in this example.

```bash
kubectl apply -f deployment_from_bentorequest.yaml
```

Once the resources are created, the `yatai-image-builder` operator will reconcile the `BentoRequest` resource and spawn a pod to download and build the container image from the provided Bento defined in the resource. The `yatai-image-builder` operator will push the built image to the container registry specified during the installation and create a `Bento` resource with the same name. At the same time, the `yatai-deployment` operator will reconcile the `BentoDeployment` resource with the provided name and create Kubernetes deployments of API Servers and Runners from the container image specified in the `Bento` resource.

### Deploym with Bento CRD

In this workflow, we'll create an OCI image from a Bento and upload the image to a container repository. After that, we'll use the `yatai-deployment` operator to deploy the Bento OCI image.

Containerize the image through `containerize` sub-command.

```bash
bentoml containerize fraud_detection:o5smnagbncigycvj -t your-username/fraud_detection:o5smnagbncigycvj
```

Push the containerized Bento image to a remote repository of your choice.

```bash
docker push your-username/fraud_detection:o5smnagbncigycvj
```

Apply the `Bento` and `BentoDeployment` resources as defined in `deployment_from_bento.yaml` file included in this example.

```bash
kubectl apply -f deployment_from_bento.yaml
```

Once the resources are created, the `yatai-deployment` operator will reconcile the `BentoDeployment` resource with the provided name and create Kubernetes deployments of API Servers and Runners from the container image specified in the `Bento` resource.

## Verify Deployment

Verify the deployment of API Servers and Runners. Note that API server and runners are run in separate pods and created in separate deployments that can be scaled independently.

```bash
kubectl -n kubeflow get pods -l yatai.ai/bento-deployment=fraud-detection

NAME                                        READY   STATUS    RESTARTS   AGE
fraud-detection-67f84686c4-9zzdz            4/4     Running   0          10s
fraud-detection-runner-0-86dc8b5c57-q4c9f   3/3     Running   0          10s
fraud-detection-runner-1-846bdfcf56-c5g6m   3/3     Running   0          10s
fraud-detection-runner-2-6d48794b7-xws4j    3/3     Running   0          10s
```

![image](https://raw.githubusercontent.com/bentoml/BentoML/main/docs/source/_static/img/kubeflow-fraud-detection.png)

Port forward the Fraud Detection service to test locally. You should be able to visit the Swagger page of the service by requesting http://0.0.0.0:8080 while port forwarding.

```bash
kubectl -n kubeflow port-forward svc/fraud-detection 8080:3000 --address 0.0.0.0
```

## Conclusion

Congratulations! You completed the example. Let's recap what we have learned.

- Trained three fraud detection models and saved them to the BentoML model store.
- Created a BentoML service that runs inferences on all three models simultaneously, combines and returns the results.
- Containerized the BentoML service into an OCI image and pushed the image to a remote repository.
- Created BentoML CRDs on Kubernetes to deploy the Bento in a microservice architecture.
