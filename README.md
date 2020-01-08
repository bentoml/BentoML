[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://bentoml.readthedocs.io/en/latest/?badge=latest)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

> From ML model to production API endpoint with a few lines of code


[![BentoML](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml.png)](https://github.com/bentoml/BentoML)

[Getting Started](https://github.com/bentoml/BentoML#getting-started) | [Documentation](http://bentoml.readthedocs.io) | [Gallery](https://github.com/bentoml/gallery) | [Contributing](https://github.com/bentoml/BentoML#contributing) | [Releases](https://github.com/bentoml/BentoML#releases) | [License](https://github.com/bentoml/BentoML/blob/master/LICENSE) | [Blog](https://medium.com/bentoml)


BentoML makes it easy to __serve and deploy machine learning models__ in the cloud.

It is an open source framework for machine learning teams to build cloud-native prediction API
services that are ready for production. BentoML supports most popular ML training frameworks
and common deployment platforms including major cloud providers and docker/kubernetes.

👉 [Join BentoML Slack community](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
 to hear about the latest development updates.

---


## Getting Started

Installation with pip:
```bash
pip install bentoml
```

Defining a prediction service with BentoML:

```python
import bentoml
from bentoml.handlers import DataframeHandler
from bentoml.artifact import SklearnModelArtifact

@bentoml.env(pip_dependencies=["scikit-learn"]) # defining pip/conda dependencies to be packed
@bentoml.artifacts([SklearnModelArtifact('model')]) # defining required artifacts, typically trained models
class IrisClassifier(bentoml.BentoService):

    @bentoml.api(DataframeHandler) # defining prediction service endpoint and expected input format
    def predict(self, df):
        # Pre-processing logic and access to trained model artifacts in API function
        return self.artifacts.model.predict(df)
```

Train a classifier model with default Iris dataset and pack the trained model
with the BentoService `IrisClassifier` defined above:

```python
from sklearn import svm
from sklearn import datasets

if __name__ == "__main__":
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    # Create a iris classifier service
    iris_classifier_service = IrisClassifier()

    # Pack it with the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to a BentoService bundle
    saved_path = iris_classifier_service.save()
```

A BentoService bundle is a versioned file archive, containing the BentoService you
defined, along with trained model artifacts, dependencies and configurations.

Now you can start a REST API server based off the saved BentoService bundle form
command line:
```bash
bentoml serve {saved_path}
```

If you are doing this only local machine, visit [http://127.0.0.1:5000](http://127.0.0.1:5000)
in your browser to play around with the API server's Web UI for debugging and
sending test request. You can also send prediction request with `curl` from command line:

```bash
curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data '[[5.1, 3.5, 1.4, 0.2]]' \
  http://localhost:5000/predict
```

Saved BentoService bundle is also structured to work as a docker build context, which can be
used to build a docker image for deployment:
```bash
docker build -t my_api_server {saved_path}
```

The saved BentoService bundle can also be loaded directly from command line:
```bash
bentoml predict {saved_path} --input='[[5.1, 3.5, 1.4, 0.2]]'

# alternatively:
bentoml predict {saved_path} --input='./iris_test_data.csv'
```

The saved bundle is pip-installable and can be directly distributed as a PyPI package:
```bash
pip install {saved_path}
```
```python
# Your BentoService class name will become packaged name
import IrisClassifier

installed_svc = IrisClassifier.load()
installed_svc.predict([[5.1, 3.5, 1.4, 0.2]])
```

Deploy the saved BentoService to cloud services such as AWS Lambda with the `bentoml `command:
```
bentoml deployment create my-iris-classifier --bento IrisClassifier:{VERSION} --platform=aws-lambda
```

To learn more, try out our 5-mins Quick Start notebook using BentoML to turn a trained sklearn model into a containerized REST API server, and then deploy it to AWS Lambda: [Download](https://github.com/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb), [Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/bentoml/bentoml/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb)


## Examples

#### FastAI

* Pet Image Classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb)
* Salary Range Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb)


#### Scikit-Learn

* Sentiment Analysis - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb)


#### PyTorch

* Fashion MNIST - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb)
* CIFAR-10 Image Classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb)


#### Tensorflow Keras

* Fashion MNIST - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb)
* Text Classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb)
* Toxic Comment Classifier - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb)

### Tensorflow 2.0

* tf.Function model - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb)


#### XGBoost

* Titanic Survival Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb)
* League of Legend win Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb)

#### LightGBM

* Titanic Survival Prediction -  [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb)


#### H2O

* Loan Default Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb)
* Prostate Cancer Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb)

 Visit [bentoml/gallery](https://github.com/bentoml/gallery) repository for more
 example projects demonstrating how to use BentoML.


### Deployment guides:

- [BentoML AWS Lambda Deployment Guide](https://github.com/bentoml/BentoML/blob/master/guides/deployment/deploy-with-serverless)
- [BentoML AWS SageMaker Deployment Guide](https://github.com/bentoml/BentoML/blob/master/guides/deployment/deploy-with-sagemaker)
- [BentoML Clipper.ai Deployment Guide](https://github.com/bentoml/BentoML/blob/master/guides/deployment/deploy-with-clipper/bentoml-clipper-deployment-guide.ipynb)
- [BentoML AWS ECS Deployment Guide](https://github.com/bentoml/BentoML/tree/master/guides/deployment/deploy-with-aws-ecs)
- [BentoML Google Cloud Run Deployment Guide](https://github.com/bentoml/BentoML/blob/master/guides/deployment/deploy-with-google-cloud-run/deploy-with-google-cloud-run.ipynb)
- [BentoML Kubernetes Deployment Guide](https://github.com/bentoml/BentoML/tree/master/guides/deployment/deploy-with-kubernetes)


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
  with new or custom frameworks

* __Deploy Anywhere__ - BentoService bundle can be easily deployed with
  platforms such as [Docker](https://www.docker.com/),
  [Kubernetes](https://kubernetes.io/),
  [Serverless](https://github.com/serverless/serverless),
  [Airflow](https://airflow.apache.org) and [Clipper](http://clipper.ai),
  on cloud platforms including AWS, Google Cloud, and Azure

* __Custom Runtime Backend__ - Easily integrate your python pre-processing code with
  high-performance deep learning runtime backend, such as
  [tensorflow-serving](https://github.com/tensorflow/serving)

* __Workflow Designed For Teams__ - The YataiService component in BentoML provides
  Web UI and APIs for managing and deploying all the models and prediction services
  your team has created or deployed, in a centralized service.


## Documentation

Full documentation and API references can be found at [bentoml.readthedocs.io](http://bentoml.readthedocs.io)


## Usage Tracking

BentoML library by default reports basic usages using
[Amplitude](https://amplitude.com). It helps BentoML authors to understand how
people are using this tool and improve it over time. You can easily opt-out by
running the following command from terminal:

```bash
bentoml config set usage_tracking=false
```

## Contributing

Have questions or feedback? Post a [new github issue](https://github.com/bentoml/BentoML/issues/new/choose)
or discuss in our Slack channel: [![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

Want to help build BentoML? Check out our
[contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md) and the
[development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md).

## Releases

BentoML is under active development and is evolving rapidly. **Currently it is a
Beta release, we may change APIs in future releases**.

Read more about the latest features and changes in BentoML from the [releases page](https://github.com/bentoml/BentoML/releases).


## License

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large)
