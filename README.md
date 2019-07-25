[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://bentoml.readthedocs.io/en/latest/?badge=latest)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](http://bit.ly/2N5IpbB)

> From a model in jupyter notebook to production API service in 5 minutes


[![BentoML](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/_static/img/bentoml.png)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb)

[Installation](https://github.com/bentoml/BentoML#installation) | [Getting Started](https://github.com/bentoml/BentoML#getting-started) | [Documentation](http://bentoml.readthedocs.io) | [Examples](https://github.com/bentoml/BentoML#examples) | [Contributing](https://github.com/bentoml/BentoML#contributing) | [License](https://github.com/bentoml/BentoML#license)


BentoML is a python framework for serving and operating machine learning
models, making it easy to promote trained models into high performance prediction
services.

The framework provides high-level APIs for defining an ML service and packaging
its trained model artifacts, preprocessing source code, dependencies, and
configurations into a standard file format called BentoArchive - which can be
deployed as REST API model server, PyPI package, CLI tool, or batch scoring job.


Check out our 5-mins quick start notebook [![Google Colab Badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb) using BentoML to productionize a scikit-learn model and deploy it to AWS Lambda.

---


## Installation

[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)

```python
pip install bentoml
```

## Getting Started


Defining a machine learning service with BentoML is as simple as a few lines of code:


```python
%%writefile keras_fashion_mnist.py

import numpy as np
from PIL import Image
from bentoml import api, artifacts, env, BentoService
from bentoml.artifact import KerasModelArtifact
from bentoml.handlers import ImageHandler

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@env(conda_dependencies=['tensorflow', 'Pillow'])
@artifacts([KerasModelArtifact('classifier')])
class KerasFashionMnistService(BentoService):
        
    @api(ImageHandler, pilmode='L')
    def predict(self, img):
        img = Image.fromarray(img).resize((28, 28))
        img = np.array(img.getdata()).reshape((1,28,28,1))
        class_idx = self.artifacts.classifier.predict_classes(img)[0]
        return class_names[class_idx]
```

Import the defined BentoService, pack with trained model, and save a versioned
archive to file system:
```python
from keras_fashion_mnist import KerasFashionMnistService

model = keras.Sequential()
model.add(...)
...
model.compile(...)
model.fit(...)
model.evaluate(...)

# Packaging trained model for serving in production:
saved_path = KerasFashionMnistService.pack(classifier=model).save('/my_bento_archives')
```

Now you can start a REST API server based off the saved BentoArchive:
```bash
bentoml serve {saved_path}
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to play
around with the Web UI of the REST API model server, or try sending prediction
request with `curl`:

```bash
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: image/png"
--data-binary @sample_image.png
```

The saved archive can also be used directly from CLI:
```bash
bentoml predict {saved_path} --input=sample_image.png
```

Or installed as a python package:
```bash
pip install {saved_path}
```

Try out the full example notebook
[here](https://github.com/bentoml/BentoML/blob/master/examples/keras-fashion-mnist/keras-fashion-mnist-classification.ipynb).


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
  [Keras](https://keras.io/),
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

- Quick Start Guide - [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb)
- **Scikit-learn** Sentiment Analysis - [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb)
- **Keras** Text Classification - [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/keras-text-classification/keras-text-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/keras-text-classification/keras-text-classification.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/keras-text-classification/keras-text-classification.ipynb)
- **Keras** Fashion MNIST classification - [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/keras-fashion-mnist/keras-fashion-mnist-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/keras-fashion-mnist/keras-fashion-mnist-classification.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/keras-fashion-mnist/keras-fashion-mnist-classification.ipynb)
- **FastAI** Pet Classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/pet-classification/notebook.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fast-ai/pet-classification/notebook.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/fast-ai/pet-classification/notebook.ipynb)
- **FastAI** Tabular CSV - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/tabular-csv/notebook.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fast-ai/tabular-csv/notebook.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/fast-ai/tabular-csv/notebook.ipynb)
- **PyTorch** Fashion MNIST classification - [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb)
- **PyTorch** CIFAR-10 Image classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/cifar10_image_classification/notebook.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/gallery/blob/master/pytorch/cifar10_image_classification/notebook.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/pytorch/cifar10_image_classification/notebook.ipynb)
- **XGBoost** Titanic Survival Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb)
- **H2O** Classification- [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/h2o-classification/h2o-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/BentoML/blob/master/examples/h2o-classification/h2o-classification.ipynb) | [source](https://github.com/bentoml/BentoML/blob/master/examples/h2o-classification/h2o-classification.ipynb) 

More examples can be found under the
[BentoML/examples](https://github.com/bentoml/BentoML/tree/master/examples)
directory or the [bentoml/gallery](https://github.com/bentoml/gallery) repo.


Deployment guides:
- [Serverless deployment with AWS Lambda](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-serverless)
- [API server deployment with AWS SageMaker](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-sagemaker)
- [API server deployment with Clipper](https://github.com/bentoml/BentoML/blob/master/example/deploy-with-clipper/deploy-iris-classifier-to-clipper.ipynb)
- [API server deployment on Kubernetes](https://github.com/bentoml/BentoML/tree/master/examples/deploy-with-kubernetes)

## Usage Tracking

BentoML library by default reports basic usages using
[Amplitude](https://amplitude.com). It helps BentoML authors to understand how
people are using this tool and improve it over time. You can easily opt-out by
running the following command from terminal:

```bash
bentoml config set usage_tracking=false
```

Or from your python code:
```python
import bentoml
bentoml.config.set('core', 'usage_tracking', 'false')
```

We also collect example notebook page views to help us understand the community
interests. To opt-out of tracking, delete the ~~!\[Impression\]\(http...~~ line in the first
markdown cell of our example notebooks. 


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

Watch BentoML Github repo for future releases:

![gh-watch](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/_static/img/gh-watch-screenshot.png)

## License

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/master/LICENSE)


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
