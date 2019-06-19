# BentoML
> From a model in jupyter notebook to production API service in 5 minutes.

[![project status](https://www.repostatus.org/badges/latest/active.svg)](http://bentoml.ai/)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://bentoml.readthedocs.io/en/latest/?badge=latest)
[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)


BentoML is a python framework for building, shipping and running machine learning
services. It provides high-level APIs for defining an ML service and packaging
its artifacts, source code, dependencies, and configurations into a
production-system-friendly format that is ready for deployment.


[![Google Colab Badge](https://badgen.net/badge/Launch%20Quick%20Start%20Guide/on%20Google%20Colab/blue?icon=terminal)](http://bit.ly/2ID50XP)


---

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Examples](#examples)
- [Releases and Contributing](#releases-and-contributing)
- [License](#license)


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
  [xgboost](https://github.com/dmlc/xgboost) and can be easily extended to work
  with new or custom frameworks.

* __Deploy Anywhere__ - BentoML bundled ML service can be easily deployed with
  platforms such as [Docker](https://www.docker.com/),
  [Kubernetes](https://kubernetes.io/),
  [Serverless](https://github.com/serverless/serverless),
  [Airflow](https://airflow.apache.org) and [Clipper](http://clipper.ai),
  on cloud platforms including AWS, Gogole Cloud, and Azure.

* __Custom Runtime Backend__ - Easily integrate your python pre-processing code with
  high-performance deep learning runtime backend, such as
  [tensorflow-serving](https://github.com/tensorflow/serving).


## Installation

![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)
![pypi status](https://img.shields.io/pypi/v/bentoml.svg)

```python
pip install bentoml
```

Verify installation:

```bash
bentoml --version
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

Read our 5-mins [Quick Start Guide](http://bit.ly/2ID50XP),
showcasing how to productionize a scikit-learn model and deploy it to AWS Lambda.


## Documentation

Official BentoML documentation can be found at [bentoml.readthedocs.io](http://bentoml.readthedocs.io)


## Examples

All examples can be found under the
[BentoML/examples](https://github.com/bentoml/BentoML/tree/master/examples)
directory. More tutorials and examples coming soon!

- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2ID50XP) - [Quick Start Guide](https://github.com/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2KegK6n) - [Scikit-learn Sentiment Analysis](https://github.com/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2KdwNRN) - [H2O Classification](https://github.com/bentoml/BentoML/blob/master/examples/h2o-classification/h2o-classification.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2IbtfNO) - [Keras Text Classification](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-text-classification/tf-keras-text-classification.ipynb)
- [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](http://bit.ly/2wPh3M3) - [XGBoost Titanic Survival Prediction](https://github.com/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb)
- [(WIP) PyTorch Fashion MNIST classification](https://github.com/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb)
- [(WIP) Tensorflow Keras Fashion MNIST classification](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-fashion-mnist/tf-keras-fashion-mnist-classification.ipynb)


Deployment guides:
- [Serverless deployment with AWS Lambda](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-serverless)
- [API server deployment with AWS SageMaker](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-sagemaker)
- [(WIP) API server deployment on Kubernetes](https://github.com/bentoml/BentoML/tree/master/examples/deploy-with-kubernetes)
- [(WIP) API server deployment with Clipper](https://github.com/bentoml/BentoML/pull/151)


We collect example notebook page views to help us improve this project.
To opt-out of tracking, delete the `[Impression]` line in the first markdown cell of any example notebook: ~~!\[Impression\]\(http...~~


## Releases and Contributing

BentoML is under active development and is evolving rapidly. **Currently it is a
Beta release, we may change APIs in future releases**.

To make sure you have a pleasant experience, please read the [code of conduct](https://github.com/bentoml/BentoML/blob/master/CODE_OF_CONDUCT.md).
It outlines core values and beliefs and will make working together a happier experience.

Have questions or feedback? Post a [new github issue](https://github.com/bentoml/BentoML/issues/new/choose)
or join our gitter chat room: [![join the chat at https://gitter.im/bentoml/BentoML](https://badges.gitter.im/bentoml/BentoML.svg)](https://gitter.im/bentoml/BentoML?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Want to help build BentoML? Check out our
[contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md) and the
[development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md)
for setting up local development and testing environments for BentoML.

Happy hacking!


## License

BentoML is under Apache License 2.0, as found in the LICENSE file.


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
