[<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml-readme-header.jpeg" width="600px" margin-left="-5px">](https://github.com/bentoml/BentoML)


## The easiest way to build Machine Learning APIs

_Multi-framework  /  High-performance  /  Easy to learn  /  Production ready_

[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![Actions Status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
<img src="https://static.scarf.sh/a.png?x-pxid=0beb35eb-7742-4dfb-b183-2228e8caf04c">

What does BentoML do?
* Package models trained with _any ML framework_ and reproduce them for model serving in 
    production
* Package once and _deploy anywhere_ for real-time API serving or offline batch serving
* High-Performance API model server with _adaptive micro-batching_ support
* Central storage hub with Web UI and APIs for managing and accessing packaged models
* Modular and flexible design allowing advanced users to easily customize
                                                           
BentoML is a framework for serving, managing and deploying machine learning models. It 
is aiming to bridge the gap between Data Science and DevOps, and enable data science 
teams to continuesly deliver prediction services to production. 

👉 Join the community:
[BentoML Slack Channel](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
and [BentoML Discussions](https://github.com/bentoml/BentoML/discussions).

---

- [How it works](https://github.com/bentoml/BentoML#how-bentoml-works)
- [Documentation](https://docs.bentoml.org/)
- [Getting Started](https://docs.bentoml.org/en/stable/quickstart.html)
- [Example Gallery](https://github.com/bentoml/gallery)
- [Why BentoML](https://github.com/bentoml/BentoML#why-bentoml)
- [Contributing](https://github.com/bentoml/BentoML#contributing)
- [License](https://github.com/bentoml/BentoML/blob/master/LICENSE)


## How BentoML works

BentoML provides abstractions for creating prediction service that's bundled with one
or multiple trained models. User can define inference APIs with serving logic with
Python code and specify the expected input/output data format:

```python
import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

from my_library import preprocess

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('my_model')])
class MyPredictionService(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(input=DataframeInput(orient="records"), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        model_input = preprocess(df)
        return self.artifacts.my_model.predict(model_input)
```

At the end of your model training pipeline, import your BentoML prediction service
class, pack it with your trained model, and persist the entire prediction service with
`save` call at the end:

```python
from my_prediction_service import MyPredictionService
svc = MyPredictionService()
svc.pack('my_model', my_sklearn_model)
svc.save()  # default saves to ~/bentoml/repository/MyPredictionService/{version}/
```

This will save all the code, files, serialized models, and configs required for 
reproducing this prediction service for inference. BentoML automatically find all the 
pip package dependencies and local python code dependencies, and make sure all those are
packaged and versioned with your code and model in one place.

With the saved prediction service, a user can easily start a local API server hosting it:
```bash
bentoml serve MyPredictionService:latest
```

And create a docker container image for this API model server with just one command:
```bash
bentoml containerize my_prediction_service MyPredictionService:latest -t my_prediction_service

docker run -p 5000:5000 my_prediction_service
```

BentoML will make sure the container has all the required dependencies installed. In
addition to the model inference API, this containerized BentoML model server also comes
with instrumentations, metrics/healthcheck endpoints, prediction logging, tracing and it
is thus ready for your DevOps team to deploy in production.

If you are at a small team without DevOps support, BentoML also provides an [one-click
deployment option](https://github.com/bentoml/BentoML#deployment-options), which deploys
the model server API to cloud platforms with minimum setup.

Read the [Quickstart Guide](https://docs.bentoml.org/en/latest/quickstart.html) 
to learn more about the basic functionalities of BentoML. You can also try it out 
[here on Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb).

## Documentation

BentoML documentation: [https://docs.bentoml.org/](https://docs.bentoml.org/)

* [Quickstart Guide](https://docs.bentoml.org/en/latest/quickstart.html), try it out [on Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb) 
* [Core Concepts](https://docs.bentoml.org/en/latest/concepts.html)
* [API References](https://docs.bentoml.org/en/latest/api/index.html)
* [FAQ](https://docs.bentoml.org/en/latest/faq.html)
* Example projects: [bentoml/Gallery](https://github.com/bentoml/gallery)


### Kye Features

Online serving with API model server:
* **Containerized model server** for production deployment with Docker, Kubernetes, OpenShift, AWS ECS, Azure, GCP GKE, etc
* **Adaptive micro-batching** for optimal online serving performance
* Discover and package all dependencies automatically, including PyPI, conda packages and local python modules
* Support **multiple ML frameworks** including PyTorch, TensorFlow, Scikit-Learn, XGBoost, and [many more](https://github.com/bentoml/BentoML#frameworks)
* Serve compositions of **multiple models**
* Serve **multiple endpoints** in one model server
* Serve any Python code along with trained models
* Automatically generate HTTP API spec in **Swagger/OpenAPI** format
* **Prediction logging** and feedback logging endpoint
* Health check endpoint and **Prometheus** `/metrics` endpoint for monitoring
* Load and replay historical prediction request logs (roadmap)
* Model serving via gRPC endpoint (roadmap)

Advanced workflow for model serving and deployment:
* **Central repository** for managing all your team's packaged models via Web UI and API
* Launch inference run from CLI or Python, which enables **CI/CD** testing, programmatic 
    access and **batch offline inference job**
* **One-click deployment** to cloud platforms including AWS Lambda, AWS SageMaker, and Azure Functions
* Distributed batch job or streaming job with **Apache Spark** (improved Spark support is on roadmap)
* **Advanced model deployment workflows** for Kubernetes, including auto-scaling, scale-to-zero, A/B testing, canary deployment, and multi-armed-bandit (roadmap)
* Deep integration with ML experimentation platforms including MLFlow, Kubeflow (roadmap)


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


### Deployment Options

Be sure to check out [deployment overview doc](https://docs.bentoml.org/en/latest/deployment/index.html)
to understand which deployment option is best suited for your use case.

* One-click deployment with BentoML:
  - [AWS Lambda](https://docs.bentoml.org/en/latest/deployment/aws_lambda.html)
  - [AWS SageMaker](https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html)
  - [Azure Functions](https://docs.bentoml.org/en/latest/deployment/azure_functions.html)

* Deploy with open-source platforms:
  - [Docker](https://docs.bentoml.org/en/latest/deployment/docker.html)
  - [Kubernetes](https://docs.bentoml.org/en/latest/deployment/kubernetes.html)
  - [Knative](https://docs.bentoml.org/en/latest/deployment/knative.html)
  - [Kubeflow](https://docs.bentoml.org/en/latest/deployment/kubeflow.html)
  - [KFServing](https://docs.bentoml.org/en/latest/deployment/kfserving.html)
  - [Clipper](https://docs.bentoml.org/en/latest/deployment/clipper.html)

* Deploy directly to cloud services:
  - [AWS ECS](https://docs.bentoml.org/en/latest/deployment/aws_ecs.html)
  - [Google Cloud Run](https://docs.bentoml.org/en/latest/deployment/google_cloud_run.html)
  - [Azure container instance](https://docs.bentoml.org/en/latest/deployment/azure_container_instance.html)
  - [Heroku](https://docs.bentoml.org/en/latest/deployment/heroku.html)


## Why BentoML

Moving trained Machine Learning models to serving applications in production is hard. It
is a sequential process across data science, engineering and DevOps teams: after a
model is trained by the data science team, they hand it over to the engineering team to
refine and optimize code and creates an API, before DevOps can deploy.

And most importantly, Data Science teams want to continously repeat this process,
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
