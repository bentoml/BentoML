[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![Actions Status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
<img src="https://static.scarf.sh/a.png?x-pxid=0beb35eb-7742-4dfb-b183-2228e8caf04c">

[![BentoML](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml.png)](https://github.com/bentoml/BentoML)



BentoML is an open-source framework for __ML model serving__, bridging the gap between
Data Science and DevOps.

What does BentoML do?
* Package models trained with any framework and reproduce them for model serving in 
    production
* Package once and deploy anywhere, supporting Docker, Kubernetes, Apache Spark,
    Airflow, Kubeflow, Knative, AWS Lambda, SageMaker, Azure ML, GCP, Heroku and more
* High-Performance API model server with adaptive micro-batching support
* Central hub for teams to manage and access packaged models via Web UI and API

👉 To connect with the community and ask questions, take a look at 
[BentoML Discussions](https://github.com/bentoml/BentoML/discussions) on Github and
the [BentoML Slack Community](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg).


---

- [Why BentoML](https://github.com/bentoml/BentoML#why-bentoml)
- [Getting Started](https://github.com/bentoml/BentoML#getting-started)
- [Documentation](https://docs.bentoml.org/)
- [Gallery](https://github.com/bentoml/gallery)
- [Contributing](https://github.com/bentoml/BentoML#contributing)
- [Releases](https://github.com/bentoml/BentoML#releases)
- [License](https://github.com/bentoml/BentoML/blob/master/LICENSE)


## Why BentoML

Getting Machine Learning models into production is hard. Data Scientists are not experts
in building production services and DevOps best practices. The trained models produced
by a Data Science team are hard to test and hard to deploy. This often leads us to a
time consuming and error-prone workflow, where a jupyter notebook along with pickled or
weights file is handed over to ML engineers, in order to rebuild a model server that
can be deployed and managed by DevOps.

BentoML is framework for ML model serving, making it possible for Data Scientists to
create production-ready prediction services with a set of high-level APIs, without them
worrying about the infrastructure integrations and performance optimizations. BentoML
does all those under the hood, which allows DevOps to seamlessly work with Data Science
team, helping to deploy and operate their models, packaged in the BentoML format.

Check out [Frequently Asked Questions](https://docs.bentoml.org/en/latest/faq.html) page
on how does BentoML compares to Tensorflow-serving, Clipper, AWS SageMaker, MLFlow, etc.

<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml-overview.png" width="600">


## Getting Started

Run this Getting Started guide on Google Colab: [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb) 

BentoML requires python 3.6 or above, install with `pip`:
```bash
pip install bentoml
```

Before starting, let's prepare a trained model for serving with BentoML.

Install required dependencies to run the example code:
```
pip install scikit-learn pandas
```

Train a classifier model on the [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set):
```python
from sklearn import svm
from sklearn import datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)
```

A minimal prediction service in BentoML looks something like this:

```python
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact

@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):

    @api(input=DataframeInput())
    def predict(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)
```

This code defines a prediction service that packages a scikit-learn model and provides
an inference API that expects input data of `pandas.Dataframe` type. The user-defined
API function `predict` defines how the input dataframe data will be processed and feeded
to the scikit-learn model being packaged. BentoML also supports other API input 
types such as `JsonInput`, `ImageInput`, `FileInput` and 
[more](https://docs.bentoml.org/en/latest/api/adapters.html).

The following code packages the trained model with the prediction service class
`IrisClassifier` defined above, and then saves the IrisClassifier instance to disk 
in the BentoML format:

```python
from iris_classifier import IrisClassifier

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()
```

BentoML stores all packaged model files under the
`~/bentoml/{service_name}/{service_version}` directory by default.
The BentoML file format contains all the code, files, and configs required to 
deploy the model for serving.

To start a REST API model server with the `IrisClassifier` saved above, use 
the `bentoml serve` command:

```bash
bentoml serve IrisClassifier:latest
```

The `IrisClassifier` model is now served at `localhost:5000`. Use `curl` command to send
a prediction request:

```bash
$ curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data '[[5.1, 3.5, 1.4, 0.2]]' \
  http://localhost:5000/predict
```

Note that BentoML API server automatically converts the Dataframe JSON format into a
`pandas.DataFrame` object before sending it to the user-defined inference API function.

The BentoML API server also provides a simple web UI dashboard.
Go to http://localhost:5000 in the browser and use the Web UI to send
prediction request:

<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bento-api-server-web-ui.png" width="700">

One common way of distributing this model API server for production deployment, is via
Docker containers. And BentoML provides a convenient way to do that.

If you already have docker configured, simply run the follow command to product a 
docker container serving the `IrisClassifier` prediction service created above:

```bash
$ bentoml containerize IrisClassifier:latest -t iris-classifier
```

Start the docker container to test out its functionalities:
```bash
$ docker run -p 5000:5000 iris-classifier --enable-microbatch --workers=1
```


## Documentation

BentoML full documentation: [https://docs.bentoml.org/](https://docs.bentoml.org/)

- Quick Start Guide: [https://docs.bentoml.org/en/latest/quickstart.html](https://docs.bentoml.org/en/latest/quickstart.html)
- Core Concepts: [https://docs.bentoml.org/en/latest/concepts.html](https://docs.bentoml.org/en/latest/concepts.html)
- Deployment Guides: https://docs.bentoml.org/en/latest/deployment/index.html
- API References: [https://docs.bentoml.org/en/latest/api/index.html](https://docs.bentoml.org/en/latest/api/index.html)
- Frequently Asked Questions: [https://docs.bentoml.org/en/latest/faq.html](https://docs.bentoml.org/en/latest/faq.html)


### Frameworks

BentoML supports these ML frameworks out-of-the-box:

* Scikit-Learn - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#scikit-learn) | [Examples](https://github.com/bentoml/gallery#scikit-learn)
* PyTorch - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#pytorch) | [Examples](https://github.com/bentoml/gallery#pytorch)
* Tensorflow 2 - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#tensorflow-2-0) | [Examples](https://github.com/bentoml/gallery#tensorflow-20)
* Tensorflow Keras - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#tensorflow-keras) | [Examples](https://github.com/bentoml/gallery#tensorflow-keras)
* XGBoost - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#xgboost) | [Examples](https://github.com/bentoml/gallery#xgboost)
* LightGBM - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#lightgbm) | [Examples](https://github.com/bentoml/gallery#lightgbm)
* FastText - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#fasttext) | [Examples](https://github.com/bentoml/gallery#fasttext)
* FastAI - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#fastai) | [Examples](https://github.com/bentoml/gallery#fastai)
* H2O - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#h2o) | [Examples](https://github.com/bentoml/gallery#h2o)
* ONNX - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#onnx) | [Examples](https://github.com/bentoml/gallery#onnx)
* CoreML - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#coreml)
* Spacy - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#spacy)

### Examples Gallery

Visit [bentoml/gallery](https://github.com/bentoml/gallery) repository for list of
example ML projects built with BentoML.

### Deployment guides:
* Automated deployment with BentoML
  - [AWS Lambda Deployment Guide](https://docs.bentoml.org/en/latest/deployment/aws_lambda.html)
  - [AWS SageMaker Deployment Guide](https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html)
  - [Azure Functions Deployment Guide](https://docs.bentoml.org/en/latest/deployment/azure_functions.html)

* Deploy with open-source platforms:
  - [Kubernetes Deployment](https://docs.bentoml.org/en/latest/deployment/kubernetes.html)
  - [Knative Deployment](https://docs.bentoml.org/en/latest/deployment/knative.html)
  - [Kubeflow Deployment](https://docs.bentoml.org/en/latest/deployment/kubeflow.html)
  - [KFServing Deployment](https://docs.bentoml.org/en/latest/deployment/kfserving.html)
  - [Clipper.ai Deployment Guide](https://docs.bentoml.org/en/latest/deployment/clipper.html)

* Deploy with cloud services:
  - [AWS ECS Deployment](https://docs.bentoml.org/en/latest/deployment/aws_ecs.html)
  - [Google Cloud Run Deployment](https://docs.bentoml.org/en/latest/deployment/google_cloud_run.html)
  - [Azure container instance Deployment](https://docs.bentoml.org/en/latest/deployment/azure_container_instance.html)
  - [Heroku Deployment](https://docs.bentoml.org/en/latest/deployment/heroku.html)


## Contributing

Have questions or feedback? Post a [new github issue](https://github.com/bentoml/BentoML/issues/new/choose)
or discuss in our Slack channel: [![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)


Want to help build BentoML? Check out our
[contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md) and the
[development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md).


[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/0)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/0)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/1)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/1)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/2)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/2)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/3)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/3)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/4)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/4)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/5)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/5)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/6)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/6)[![](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/images/7)](https://sourcerer.io/fame/yubozhao/bentoml/BentoML/links/7)


## Releases

BentoML is under active development and is evolving rapidly.
Currently it is a Beta release, __we may change APIs in future releases__.

Read more about the latest features and changes in BentoML from the [releases page](https://github.com/bentoml/BentoML/releases).


## Usage Tracking

BentoML by default collects anonymous usage data using [Amplitude](https://amplitude.com).
It only collects BentoML library's own actions and parameters, no user or model data will be collected.
[Here is the code that does it](https://github.com/bentoml/BentoML/blob/master/bentoml/utils/usage_stats.py).

This helps BentoML team to understand how the community is using this tool and
what to build next. You can easily opt-out of usage tracking by running the following
command:

```bash
# From terminal:
bentoml config set usage_tracking=false
```

```python
# From python:
import bentoml
bentoml.config().set('core', 'usage_tracking', 'False')
```


## License

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
