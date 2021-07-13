[<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)

## Model Serving Made Easy  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20Machine%20Learning%20Model%20Serving%20Made%20Easy%20&url=https://github.com/bentoml/BentoML&via=bentomlai&hashtags=mlops,modelserving,ML,AI,machinelearning,bentoml)

BentoML is a flexible, high-performance framework for serving, managing, and deploying machine learning models. 

* Supports __multiple ML frameworks__, including Tensorflow, PyTorch, Keras, XGBoost
  and [more](https://github.com/bentoml/BentoML#ml-frameworks)
* __Cloud native deployment__ with Docker, Kubernetes, AWS, Azure and
  [many more](https://github.com/bentoml/BentoML#deployment-options)
* __High-Performance__ online API serving and offline batch serving
* Web dashboards and APIs for model registry and deployment management

                                                           
BentoML bridges the gap between Data Science and DevOps. By providing a
standard interface for describing a prediction service, BentoML 
abstracts away how to run model inference efficiently and how model 
serving workloads can integrate with cloud infrastructures.
[See how it works!](https://github.com/bentoml/BentoML#introduction)


Join our community
[on Slack](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg) 👈

---

[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![Actions Status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)


- [Documentation](https://github.com/bentoml/BentoML#documentation)
- [Key Features](https://github.com/bentoml/BentoML#key-features)
- [Introduction](https://github.com/bentoml/BentoML#Introduction)
- [Why BentoML](https://github.com/bentoml/BentoML#why-bentoml)
- [Contributing](https://github.com/bentoml/BentoML#contributing)
- [License](https://github.com/bentoml/BentoML#license)


## Documentation

BentoML documentation: [https://docs.bentoml.org/](https://docs.bentoml.org/)

* [Quickstart Guide](https://docs.bentoml.org/en/latest/quickstart.html), try it out [on Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb) 
* [Core Concepts](https://docs.bentoml.org/en/latest/concepts.html)
* [API References](https://docs.bentoml.org/en/latest/api/index.html)
* [FAQ](https://docs.bentoml.org/en/latest/faq.html)
* Example projects: [bentoml/Gallery](https://github.com/bentoml/gallery)


### Key Features

Production-ready online serving:
* Support **multiple ML frameworks** including PyTorch, TensorFlow, Scikit-Learn, XGBoost, and [many more](https://github.com/bentoml/BentoML#ml-frameworks)
* **Containerized model server** for production deployment with Docker, Kubernetes, OpenShift, AWS ECS, Azure, GCP GKE, etc
* **Adaptive micro-batching** for optimal online serving performance
* Discover and package all dependencies automatically, including PyPI, conda packages and local python modules
* Serve compositions of **multiple models**
* Serve **multiple endpoints** in one model server
* Serve any Python code along with trained models
* Automatically generate REST API spec in **Swagger/OpenAPI** format
* **Prediction logging** and feedback logging endpoint
* Health check endpoint and **Prometheus** `/metrics` endpoint for monitoring

Standardize model serving and deployment workflow for teams:
* **Central repository** for managing all your team's prediction services  via
  Web UI and API
* Launch offline batch inference job from CLI or Python
* **One-click deployment** to cloud platforms including AWS EC2, AWS Lambda, AWS SageMaker, and Azure Functions
* Distributed batch or streaming serving with **Apache Spark**
* Utilities that simplify CI/CD pipelines for ML
* Automated offline batch inference job with **Dask** (roadmap)
* **Advanced model deployment** for Kubernetes ecosystem (roadmap)
* Integration with training and experimentation management products including MLFlow, Kubeflow (roadmap)


### ML Frameworks

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
* Spacy - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#spacy) | [Examples](https://github.com/bentoml/gallery#spacy)
* Statsmodels - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#statsmodels) | [Examples](https://github.com/bentoml/gallery#statsmodels)
* CoreML - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#coreml)
* Transformers - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#transformers)
* Gluon - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#gluon)
* Detectron - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#detectron)
* PaddlePaddle - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#paddle) | [Example](https://github.com/bentoml/gallery#paddlepaddle)
* EvalML - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#evalml)
* EasyOCR -[Docs](https://docs.bentoml.org/en/latest/frameworks.html#easyocr)
* ONNX-MLIR - [Docs](https://docs.bentoml.org/en/latest/frameworks.html#onnx-mlir)


### Deployment Options

Be sure to check out [deployment overview doc](https://docs.bentoml.org/en/latest/deployment/index.html)
to understand which deployment option is best suited for your use case.

* One-click deployment with BentoML:
  - [AWS Lambda](https://docs.bentoml.org/en/latest/deployment/aws_lambda.html)
  - [AWS SageMaker](https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html)
  - [AWS EC2](https://docs.bentoml.org/en/latest/deployment/aws_ec2.html)
  - [Azure Functions](https://docs.bentoml.org/en/latest/deployment/azure_functions.html)

* Deploy with open-source platforms:
  - [Docker](https://docs.bentoml.org/en/latest/deployment/docker.html)
  - [Kubernetes](https://docs.bentoml.org/en/latest/deployment/kubernetes.html)
  - [Knative](https://docs.bentoml.org/en/latest/deployment/knative.html)
  - [Kubeflow](https://docs.bentoml.org/en/latest/deployment/kubeflow.html)
  - [KFServing](https://docs.bentoml.org/en/latest/deployment/kfserving.html)
  - [Clipper](https://docs.bentoml.org/en/latest/deployment/clipper.html)

* Manual cloud deployment guides:
  - [AWS ECS](https://docs.bentoml.org/en/latest/deployment/aws_ecs.html)
  - [Google Cloud Run](https://docs.bentoml.org/en/latest/deployment/google_cloud_run.html)
  - [Google Cloud AI Platform Unified](https://docs.bentoml.org/en/latest/deployment/google_cloud_ai_platform.html)
  - [Azure container instance](https://docs.bentoml.org/en/latest/deployment/azure_container_instance.html)
  - [Heroku](https://docs.bentoml.org/en/latest/deployment/heroku.html)


## Introduction

BentoML provides APIs for defining a prediction service, a servable model so to speak, 
which includes the trained ML model itself, plus its pre-processing, post-processing
code, input/output specifications and dependencies. Here's what a simple
prediction service look like in BentoML: 


```python
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact

# BentoML packages local python modules automatically for deployment
from my_ml_utils import my_encoder

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('my_model')])
class MyPredictionService(BentoService):
    """
    A simple prediction service exposing a Scikit-learn model
    """

    @api(input=DataframeInput(), output=JsonOutput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` that takes tabular data in pandas.DataFrame 
        format as input, and returns Json Serializable value as output.

        A batch API is expect to receive a list of inference input and should returns
        a list of prediction results.
        """
        model_input_df = my_encoder.fit_transform(df)
        predictions = self.artifacts.my_model.predict(model_input_df)

        return list(predictions)
```

This can be easily plugged into your model training process: import your bentoml 
prediction service class, pack it with your trained model, and call `save` to persist
the entire prediction service at the end, which creates a BentoML bundle:

```python
from my_prediction_service import MyPredictionService
svc = MyPredictionService()
svc.pack('my_model', my_sklearn_model)
svc.save()  # saves to $HOME/bentoml/repository/MyPredictionService/{version}/
```

The generated BentoML bundle is a file directory that contains all the code files, 
serialized models, and configs required for reproducing this prediction service for
inference. BentoML automatically captures all the python dependencies information and
have everything versioned and managed together in one place.

BentoML automatically generates a version ID for this bundle, and keeps track of all
bundles created under the `$HOME/bentoml` directory. With a BentoML bundle, user can 
start a local API server hosting it, either by its file path or its name and version:

```bash
bentoml serve MyPredictionService:latest

# alternatively
bentoml serve $HOME/bentoml/repository/MyPredictionService/{version}/
```

A docker container image that's ready for production deployment can be created now with
just one command:
```bash
bentoml containerize MyPredictionService:latest -t my_prediction_service:v3

docker run -p 5000:5000 my_prediction_service:v3 --workers 2
```

The container image produced will have all the required dependencies installed. Besides
the model inference API, the containerized BentoML model server also comes with
Prometheus metrics, health check endpoint, prediction logging, and tracing support
out-of-the-box. This makes it super easy for your DevOps team to incorporate your models
into production systems.

BentoML's model management component is called Yatai, it means food cart in Japanese, 
and you can think of it as where you'd store your bentos 🍱. Yatai provides CLI, Web UI,
and Python API for accessing BentoML bundles you have created, and you can start a Yatai
server for your team to manage all models on cloud storage(S3, GCS, MinIO etc) and build
CI/CD workflow around it. 
[Learn more about it here](https://docs.bentoml.org/en/latest/concepts.html#model-management).

![Yatai UI](https://github.com/bentoml/BentoML/blob/master/docs/source/_static/img/yatai-service-web-ui-repository.png)

Read the [Quickstart Guide](https://docs.bentoml.org/en/latest/quickstart.html) 
to learn more about the basic functionalities of BentoML. You can also try it out 
[here on Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb).


## Why BentoML

Moving trained Machine Learning models to serving applications in production is hard. It
is a sequential process across data science, engineering and DevOps teams: after a
model is trained by the data science team, they hand it over to the engineering team to
refine and optimize code and creates an API, before DevOps can deploy.

And most importantly, Data Science teams want to continuously repeat this process,
monitor the models deployed in production and ship new models quickly. It often takes
months for an engineering team to build a model serving & deployment solution that allow
data science teams to ship new models in a repeatable and reliable way.

BentoML is a framework designed to solve this problem. It provides high-level APIs for
Data Science team to create prediction services, abstract away DevOps'
infrastructure needs and performance optimizations in the process. This allows DevOps
team to seamlessly work with data science side-by-side, deploy and operate their models
packaged in BentoML format in production.

Check out [Frequently Asked Questions](https://docs.bentoml.org/en/latest/faq.html) page
on how does BentoML compares to Tensorflow-serving, Clipper, AWS SageMaker, MLFlow, etc.

<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml-overview.png" width="600">


## Contributing

Have questions or feedback? Post a [new github issue](https://github.com/bentoml/BentoML/issues/new/choose)
or discuss in our Slack channel: [![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

Want to help build BentoML? Check out our
[contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md) and the
[development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md).


## Releases

BentoML is under active development and is evolving rapidly.
It is currently a Beta release, __we may change APIs in future releases__ and there
are still major features being worked on.

Read more about the latest updates from the [releases page](https://github.com/bentoml/BentoML/releases).


## Usage Tracking

BentoML by default collects anonymous usage data using [Amplitude](https://amplitude.com).
It only collects BentoML library's own actions and parameters, no user or model data will be collected.
[Here is the code that does it](https://github.com/bentoml/BentoML/blob/master/bentoml/utils/usage_stats.py).

This helps BentoML team to understand how the community is using this tool and what to build next. You can
easily opt-out of usage tracking by running the BentoML commands with the `--do-not-track` option.

```bash
% bentoml [command] --do-not-track
```

or by setting the `BENTOML_DO_NOT_TRACK` environment variable to `True`.

```bash
% export BENTOML_DO_NOT_TRACK=True
```


## License

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
