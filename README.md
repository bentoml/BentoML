[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)


[![BentoML](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml.png)](https://github.com/bentoml/BentoML)



BentoML is an open-source framework for __high-performance ML model serving__.

What does BentoML do?
* Create API endpoint serving trained models with just a few lines of code
* Support all major machine learning training frameworks
* High-Performance online API serving with adaptive micro-batching support
* Model Registry for teams, providing Web UI dashboard and CLI/API access
* Flexible deployment orchestration with DevOps best practices baked-in, supporting 
 Docker, Kubernetes, Kubeflow, Knative, AWS Lambda, SageMaker, Azure ML, GCP and more

👉 To follow development updates and discussion, join the 
[Bentoml Slack community](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
and the [contributors mailing list](https://groups.google.com/forum/#!forum/bentoml).


---

- [Getting Started](https://github.com/bentoml/BentoML#getting-started)
- [Documentation](https://docs.bentoml.org/)
- [Gallery](https://github.com/bentoml/gallery)
- [Contributing](https://github.com/bentoml/BentoML#contributing)
- [Releases](https://github.com/bentoml/BentoML#releases)
- [License](https://github.com/bentoml/BentoML/blob/master/LICENSE)
- [Blog](https://medium.com/bentoml)


## Why BentoML

Getting Machine Learning models into production is hard. Data Scientists are not experts
in building production services and DevOps best practices. The trained models produced
by a Data Science team are hard to test and hard to deploy. This often leads us to a
time consuming and error-prone workflow, where a pickled model or weights file is handed
over to a software engineering team.

BentoML is an end-to-end solution for model serving, making it possible for Data Science
teams to build production-ready model serving endpoints, with common DevOps best
practices and performance optimizations baked in.

Check out [Frequently Asked Questions](https://docs.bentoml.org/en/latest/faq.html) page
on how does BentoML compares to Tensorflow-serving, Clipper, AWS SageMaker, MLFlow, etc.

<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/bentoml-overview.png" width="600">


## Getting Started

BentoML requires python 3.6 or above, install with `pip`:
```bash
pip install bentoml
```

A minimal prediction service in BentoML looks something like this:

```python
# https://github.com/bentoml/BentoML/blob/master/guides/quick-start/iris_classifier.py
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

This code defines a prediction service that bundles a scikit-learn model and provides an
API that expects input data in the form of `pandas.Dataframe`. The user-defined API
function `predict` defines how the input dataframe data will be processed and used for 
inference with the bundled scikit-learn model. BentoML also supports other API input 
types such as `ImageInput`, `JsonInput` and 
[more](https://docs.bentoml.org/en/latest/api/adapters.html).

The following code trains a scikit-learn model and packages the trained model with the
`IrisClassifier` class defined above. It then saves the IrisClassifier instance to disk 
in the BentoML SavedBundle format:

```python
# https://github.com/bentoml/BentoML/blob/master/guides/quick-start/main.py
from sklearn import svm
from sklearn import datasets

from iris_classifier import IrisClassifier

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

By default, BentoML stores SavedBundle files under the `~/bentoml` directory. Users 
can also customize BentoML to use a different directory or cloud storage like
[AWS S3](https://aws.amazon.com/s3/) and [MinIO](https://min.io/), via BentoML's
model management component [YataiService](https://docs.bentoml.org/en/latest/concepts.html#customizing-model-repository),
which provides advanced model management features including a dashboard web UI:

![BentoML YataiService Bento Repository Page](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/yatai-service-web-ui-repository.png)

![BentoML YataiService Bento Details Page](https://raw.githubusercontent.com/bentoml/BentoML/master/docs/source/_static/img/yatai-service-web-ui-repository-detail.png)

Learn more about using YataiService for model management and try out the Web UI 
[here](https://docs.bentoml.org/en/latest/concepts.html#model-management).


The BentoML SavedBundle directory contains all the code, data and configs required to 
deploy the model. To start a REST API model server with the `IrisClassifier` SavedBundle, use 
the `bentoml serve` command:

```bash
bentoml serve IrisClassifier:latest
```

The `IrisClassifier` model is now served at `localhost:5000`. Use `curl` command to send
a prediction request:

```bash
curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data '[[5.1, 3.5, 1.4, 0.2]]' \
  http://localhost:5000/predict
```

The BentoML API server also provides a web UI for accessing predictions and debugging 
the server. Visit http://localhost:5000 in the browser and use the Web UI to send
prediction request:

<img src="https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bento-api-server-web-ui.png" width="700">


BentoML provides a convenient way to containerize the model API server with Docker:

1. Find where the SavedBundle directory is created in the file system:
    * The saved path is return by the `iris_classifier_service.save()` call
    * The saved path is printed in the stdout when saving: `INFO - BentoService bundle 'IrisClassifier:20200121114004_360ECB' saved to: ...`
    * Use the `bentoml get IrisClassifier:latest` command to view all the metadata including saved path

2. Run `docker build` under the SavedBundle directory which contains a generated Dockerfile,
    which will build a docker image containing the IrisClassifier API server


```bash
# If jq command not found, install jq (the command-line JSON processor) here: https://stedolan.github.io/jq/download/
saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

# Build the docker image
docker build -t iris-classifier $saved_path

# Start a container with the image build above
docker run -p 5000:5000 iris-classifier
```

This docker images makes it possible to deploy BentoML saved bundle to container 
orchestration platforms such as
[Kubeflow](https://www.kubeflow.org/docs/components/serving/bentoml/),
[Knative](https://knative.dev/community/samples/serving/machinelearning-python-bentoml/),
[Kubernetes](https://docs.bentoml.org/en/latest/deployment/kubernetes.html), which
provides advanced model deployment features such as auto-scaling, A/B testing,
scale-to-zero, canary rollout and multi-armed bandit.


BentoML can also deploy SavedBundle directly to cloud services such as AWS Lambda or 
AWS SageMaker, with the bentoml CLI command. For a list of all deployment options with 
BentoML, check out the 
[BentoML deployment guides](https://docs.bentoml.org/en/latest/deployment/index.html).


## Documentation

BentoML full documentation: [https://docs.bentoml.org/](https://docs.bentoml.org/)

- Quick Start Guide: [https://docs.bentoml.org/en/latest/quickstart.html](https://docs.bentoml.org/en/latest/quickstart.html)
- Core Concepts: [https://docs.bentoml.org/en/latest/concepts.html](https://docs.bentoml.org/en/latest/concepts.html)
- Deployment Guides: https://docs.bentoml.org/en/latest/deployment/index.html
- API References: [https://docs.bentoml.org/en/latest/api/index.html](https://docs.bentoml.org/en/latest/api/index.html)
- Frequently Asked Questions: [https://docs.bentoml.org/en/latest/faq.html](https://docs.bentoml.org/en/latest/faq.html)


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


#### Keras (with Tensorflow 1.0)
* Fashion MNIST - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/fashion-mnist/keras-fashion-mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb)
* Text Classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/text-classification/keras-text-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb)
* Toxic Comment Classifier - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb)


#### Tensorflow 2.0 / tf.keras
* tf.Function model - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb)
* Fashion MNIST - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb)
* Movie Review Sentiment with BERT - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb)


#### XGBoost
* Titanic Survival Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb)
* League of Legend win Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb)


#### LightGBM
* Titanic Survival Prediction -  [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb)


#### H2O
* Loan Default Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb)
* Prostate Cancer Prediction - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb)


#### FastText
* Text Classification - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/fasttext/text-classification/text-classification.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fasttext/text-classification/text-classification.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/fasttext/text-classification/text-classification.ipynb)


#### ONNX
* Scikit-Learn Iris Classifier - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/onnx/sklearn-iris-classifier/SK-iris-classifier.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/onnx/sklearn-iris-classifier/SK-iris-classifier.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/onnx/sklearn-iris-classifier/SK-iris-classifier.ipynb)
* ResNet50 Image recognition (ONNX model zoo) - [Google Colab](https://colab.research.google.com/github/bentoml/gallery/blob/master/onnx/resnet50/resnet50.ipynb) | [nbviewer](https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/onnx/resnet50/resnet50.ipynb) | [source](https://github.com/bentoml/gallery/blob/master/onnx/resnet50/resnet50.ipynb)


### Deployment guides:
* End-to-end deployment management with BentoML
  - [BentoML AWS Lambda Deployment Guide](https://docs.bentoml.org/en/latest/deployment/aws_lambda.html)
  - [BentoML AWS SageMaker Deployment Guide](https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html)

* Deployment guides for open-source platforms:
  - [BentoML Kubernetes Deployment](https://docs.bentoml.org/en/latest/deployment/kubernetes.html)
  - [BentoML Clipper.ai Deployment Guide](https://docs.bentoml.org/en/latest/deployment/clipper.html)
  - [BentoML Knative Deployment](https://docs.bentoml.org/en/latest/deployment/knative.html)
  - [BentoML Kubeflow Deployment](https://docs.bentoml.org/en/latest/deployment/kubeflow.html)
  - [BentoML KFServing Deployment](https://docs.bentoml.org/en/latest/deployment/kfserving.html)

* Deployment guides for Cloud service providers:
  - [BentoML AWS ECS Deployment](https://docs.bentoml.org/en/latest/deployment/aws_ecs.html)
  - [BentoML Google Cloud Run Deployment](https://docs.bentoml.org/en/latest/deployment/google_cloud_run.html)
  - [BentoML Azure container instance Deployment](https://docs.bentoml.org/en/latest/deployment/azure_container_instance.html)
  - [BentoML Heroku Deployment](https://docs.bentoml.org/en/latest/deployment/heroku.html)


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
