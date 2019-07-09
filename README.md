[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://bentoml.readthedocs.io/en/latest/?badge=latest)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](http://bit.ly/2N5IpbB)

> From a model in jupyter notebook to production API service in 5 minutes

# BentoML

[Installation](https://github.com/bentoml/BentoML#installation) | [Getting Started](https://github.com/bentoml/BentoML#getting-started) | [Documentation](http://bentoml.readthedocs.io) | [Examples](https://github.com/bentoml/BentoML#examples) | [Contributing](https://github.com/bentoml/BentoML#contributing) | [License](https://github.com/bentoml/BentoML#license)


BentoML is a python framework for building, shipping and running machine learning
services. It provides high-level APIs for defining an ML service and packaging
its artifacts, source code, dependencies, and configurations into a
production-system-friendly format that is ready for deployment.

Use BentoML if you need to:

* Turn your ML model into REST API server, Serverless endpoint, PyPI package, or CLI tool

* Manage the workflow of creating and deploying a ML service

---


## Installation

![pypi status](https://img.shields.io/pypi/v/bentoml.svg)

```python
pip install bentoml
```


## Getting Started

Defining a machine learning service with BentoML is as simple as a few lines of code:

```python
@artifacts([PickleArtifact('model')])
@env(conda_pip_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.model.predict(df)
```


[![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2ID50XP) - Try out
this 5-mins getting started guide, using BentoML to productionize a scikit-learn model and deploy it to AWS Lambda.


## Feature Highlights

* __Multiple Distribution Format__ - Easily package your Machine Learning models
  and preprocessing code into a format that works best with your inference scenario:
  * Docker Image - deploy as containers running REST API Server
  * PyPI Package - integrate into your python applications seamlessly
  * CLI tool - put your model into Airflow DAG or CI/CD pipeline
  * Spark UDF - run batch serving on a large dataset with Spark
  * Serverless Function - host your model on serverless platforms such as AWS Lambda

* __Multiple Framework Support__ - BentoML supports a wide range of ML frameworks
  out-of-the-box including [Tensorflow](https://github.com/tensorflow/tensorflow/),
  [PyTorch](https://github.com/pytorch/pytorch),
  [Scikit-Learn](https://github.com/scikit-learn/scikit-learn),
  [xgboost](https://github.com/dmlc/xgboost),
  [H2O](https://github.com/h2oai/h2o-3),
  [FastAI](https://github.com/fastai/fastai) and can be easily extended to work
  with new or custom frameworks.

* __Deploy Anywhere__ - BentoML bundled ML service can be easily deployed with
  platforms such as [Docker](https://www.docker.com/),
  [Kubernetes](https://kubernetes.io/),
  [Serverless](https://github.com/serverless/serverless),
  [Airflow](https://airflow.apache.org) and [Clipper](http://clipper.ai),
  on cloud platforms including AWS, Google Cloud, and Azure.

* __Custom Runtime Backend__ - Easily integrate your python pre-processing code with
  high-performance deep learning runtime backend, such as
  [tensorflow-serving](https://github.com/tensorflow/serving).


## Documentation

Full documentation and API references can be found at [bentoml.readthedocs.io](http://bentoml.readthedocs.io)


## Examples

All examples can be found under the
[BentoML/examples](https://github.com/bentoml/BentoML/tree/master/examples)
directory. More tutorials and examples coming soon!

- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2ID50XP) - [Quick Start Guide](https://github.com/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2KegK6n) - [Scikit-learn Sentiment Analysis](https://github.com/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2KdwNRN) - [H2O Classification](https://github.com/bentoml/BentoML/blob/master/examples/h2o-classification/h2o-classification.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2IbtfNO) - [Keras Text Classification](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-text-classification/tf-keras-text-classification.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2wPh3M3) - [XGBoost Titanic Survival Prediction](https://github.com/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/pet-classification/notebook.ipynb) - [FastAI Pet Classification](https://github.com/bentoml/gallery/blob/master/fast-ai/pet-classification/notebook.ipynb)
- [(WIP) PyTorch Fashion MNIST classification](https://github.com/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb)
- [(WIP) Tensorflow Keras Fashion MNIST classification](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-fashion-mnist/tf-keras-fashion-mnist-classification.ipynb)


Deployment guides:
- [Serverless deployment with AWS Lambda](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-serverless)
- [API server deployment with AWS SageMaker](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-sagemaker)
- [(WIP) API server deployment on Kubernetes](https://github.com/bentoml/BentoML/tree/master/examples/deploy-with-kubernetes)
- [(WIP) API server deployment with Clipper](https://github.com/bentoml/BentoML/pull/151)


We collect example notebook page views to help us improve this project.
To opt-out of tracking, delete the `[Impression]` line in the first markdown cell of any example notebook: ~~!\[Impression\]\(http...~~


## Contributing

Have questions or feedback? Post a [new github issue](https://github.com/bentoml/BentoML/issues/new/choose)
or join our Slack chat room: [![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](http://bit.ly/2N5IpbB)

Want to help build BentoML? Check out our
[contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md) and the
[development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md).

To make sure you have a pleasant experience, please read the [code of conduct](https://github.com/bentoml/BentoML/blob/master/CODE_OF_CONDUCT.md).
It outlines core values and beliefs and will make working together a happier experience.

Happy hacking!

## Releases

BentoML is under active development and is evolving rapidly. **Currently it is a
Beta release, we may change APIs in future releases**.

Read more about the latest features and changes in BentoML from the [releases page](https://github.com/bentoml/BentoML/releases).
and follow the [BentoML Community Calendar](http://bit.ly/2XvUiM2).


## License

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/master/LICENSE)


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
