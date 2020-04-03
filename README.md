[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

> From ML model to production API endpoint with a few lines of code

[![BentoML](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml.png)](https://github.com/bentoml/BentoML)

BentoML is an open-source platform for __machine learning model serving__.

What does BentoML do?

* Turn your ML model into production API endpoint with just a few lines of code
* Support all major machine learning training frameworks
* High performance API serving system with adaptive micro-batching support
* DevOps best practices baked in, simplify the transition from model development to production
* Model management for teams, providing CLI and Web UI dashboard
* Flexible model deployment orchestration with support for AWS Lambda, SageMaker, EC2, Docker, Kubernetes, KNative and more

👉 [Join BentoML Slack community](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
 to hear about the latest development updates.

---

- [Getting Started](https://github.com/bentoml/BentoML#getting-started)
- [Documentation](https://docs.bentoml.org/)
- [Gallery](https://github.com/bentoml/gallery)
- [Contributing](https://github.com/bentoml/BentoML#contributing)
- [Releases](https://github.com/bentoml/BentoML#releases)
- [License](https://github.com/bentoml/BentoML/blob/master/LICENSE)
- [Blog](https://medium.com/bentoml)


## Getting Started

Installing BentoML with `pip`:
```bash
pip install bentoml
```

A minimal prediction service in BentoML looks something like this:

```python
from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import DataframeHandler
from bentoml.artifact import SklearnModelArtifact

@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.model.predict(df)
```

This code defines a prediction service that requires a scikit-learn model, and asks
BentoML to figure out the required PyPI pip packages automatically. It also defined
an API, which is the entry point for accessing this prediction service. And the API is
expecting a `pandas.DataFrame` object as its input data.

Now you are ready to train a model and serve the model with the `IrisClassifier` service
defined above:

```python
from sklearn import svm
from sklearn import datasets

if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    # Create a iris classifier service instance
    iris_classifier_service = IrisClassifier()

    # Pack the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to disk for model serving
    saved_path = iris_classifier_service.save()
```

You've just created a BentoService SavedBundle, it's a versioned file archive that is
ready for production deployment. It contains the BentoService class you defined, all its
python code dependencies and PyPI dependencies, and the trained scikit-learn model. By
default, BentoML saves those files and related metadata under `~/bentoml` directory, but 
this is easily customizable to a different directory or cloud storage like
[Amazon S3](https://aws.amazon.com/s3/).

You can now start a REST API server by specifying the BentoService's name and version, 
or provide the file path to the saved bundle:

```bash
bentoml serve IrisClassifier:latest
# or
bentoml serve {saved_path}
```

The REST API server provides web UI for testing and debugging the server. If you are
running this command on your local machine, visit http://127.0.0.1:5000 in your browser
and try out sending API request to the server. You can also send prediction request
with `curl` from command line:

```bash
curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data '[[5.1, 3.5, 1.4, 0.2]]' \
  http://localhost:5000/predict
```

The BentoService SavedBundle directory is structured to work as a docker build context,
that can be used to build a API server docker container image:

```bash
docker build -t my-org/iris-classifier:v1 {saved_path}

docker run -p 5000:5000 -e BENTOML_ENABLE_MICROBATCH=True my-org/iris-classifier:v1
```

You can also deploy your BentoService directly to cloud services such as AWS Lambda
with bentoml CLI's deployment management commands:

```
> bentoml get IrisClassifier
BENTO_SERVICE                         CREATED_AT        APIS                       ARTIFACTS
IrisClassifier:20200121114004_360ECB  2020-01-21 19:40  predict<DataframeHandler>  model<SklearnModelArtifact>
IrisClassifier:20200120082658_4169CF  2020-01-20 16:27  predict<DataframeHandler>  clf<PickleArtifact>
...

> bentoml lambda deploy test-deploy -b IrisClassifier:20200121114004_360ECB
...

> bentoml deployment list
NAME           NAMESPACE    PLATFORM    BENTO_SERVICE                         STATUS    AGE
test-deploy    dev          aws-lambda  IrisClassifier:20200121114004_360ECB  running   2 days and 11 hours
...
```

More detailed code and walkthrough can be found in the [BentoML Quickstart Guide](https://docs.bentoml.org/en/latest/quickstart.html).

## Documentation

Full documentation and API references: [https://docs.bentoml.org/](https://docs.bentoml.org/)


## Examples

Visit [bentoml/gallery](https://github.com/bentoml/gallery) repository for more
 examples and tutorials.

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

#### Tensorflow 2.0

* tf.Function model - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb)
* Fashion MNIST - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb)


#### XGBoost

* Titanic Survival Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb)
* League of Legend win Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb)

#### LightGBM

* Titanic Survival Prediction -  [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb)


#### H2O

* Loan Default Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb)
* Prostate Cancer Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb)



### Deployment guides:

* Automated end-to-end deployment workflow with BentoML
  - [BentoML AWS Lambda Deployment Guide](https://docs.bentoml.org/en/latest/deployment/aws_lambda.html)
  - [BentoML AWS SageMaker Deployment Guide](https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html)

* Clipper Deployment
  - [BentoML Clipper.ai Deployment Guide](https://docs.bentoml.org/en/latest/deployment/clipper.html)

* Mannual Deployment
  - [BentoML AWS ECS Deployment](https://docs.bentoml.org/en/latest/deployment/aws_ecs.html)
  - [BentoML Google Cloud Run Deployment](https://docs.bentoml.org/en/latest/deployment/google_cloud_run.html)
  - [BentoML Azure container instance Deployment](https://docs.bentoml.org/en/latest/deployment/azure_container_instance.html)


## Contributing

Have questions or feedback? Post a [new github issue](https://github.com/bentoml/BentoML/issues/new/choose)
or discuss in our Slack channel: [![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)


Want to help build BentoML? Check out our
[contributing guide](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md) and the
[development guide](https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md).


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
