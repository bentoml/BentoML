# BentoML
> From a model in jupyter notebook to production API service in 5 minutes.

[![project status](https://www.repostatus.org/badges/latest/active.svg)](http://bentoml.ai/)
[![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)](https://travis-ci.org/bentoml/BentoML)
[![pypi status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)](https://travis-ci.org/bentoml/BentoML)
[![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)


BentoML is a python framework for building, shipping and running machine learning
services. It provides high-level APIs for defining an ML service and packaging
its artifacts, source code, dependencies, and configurations into a
production-system-friendly format that is ready for deployment.

Run our [Quick Start Guide](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb) in your browser!

---

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Documentation (Coming soon!)](#getting-started)
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

Let's get started with a simple scikit-learn model as an example:

```python
from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
```

To package this model with BentoML, you don't need to change anything in your
training code. Following your training workflow, create a new BentoML Service by
subclassing it:

```python
%%writefile iris_classifier.py
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import DataframeHandler

# You can also import your own python module here and BentoML will automatically
# figure out the dependency chain and package all those python modules

@artifacts([PickleArtifact('model')])
@env(conda_pip_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        # arbitrary preprocessing or feature fetching code can be placed here 
        return self.artifacts.model.predict(df)
```

The `@artifacts` decorator here tells BentoML what artifacts are required when 
packaging this BentoService. Besides `PickleArtifact`, BentoML also provides
`TfKerasModelArtifact`, `PytorchModelArtifact`, and `TfSavedModelArtifact` etc.

`@env` is designed for specifying the desired system environment in order for this
BentoService to load. Other ways you can use this decorator:

* If you already have a requirement.txt file listing all python libraries you
need:
```python
@env(requirement_txt='../myproject/requirement.txt')
```

* Or if you are running this code in a Conda environment that matches the
desired production environment:
```python
@env(with_current_conda_env=True)
```

Lastly `@api` adds an entry point for accessing this BentoService. Each
`api` will be translated into a REST endpoint when [deploying as API
server](#serving-via-rest-api), or a CLI command when [running as a CLI
tool](#use-as-cli-tool).

Each API also requires a `Handler` for defining the expected input format. In
this case, `DataframeHandler` will transform either an HTTP request or CLI
command arguments into a pandas Dataframe and pass it down to the user defined
API function. BentoML also supports `JsonHandler`, `ImageHandler` and
`TensorHandler`.

Next, to save your trained model for production use with this custom
BentoService class:

```python
# 1) import the custom BentoService defined above
from iris_classifier import IrisClassifier

# 2) `pack` it with required artifacts
svc = IrisClassifier.pack(model=clf)

# 3) save packed BentoService as archive
svc.save('./bento_archive', version='v0.0.1')
# archive will saved to ./bento_archive/IrisClassifier/v0.0.1/
```

_That's it._ You've just created your first BentoArchive. It's a directory
containing all the source code, data and configurations files required to load
and run a BentoService. You will also find three 'magic' files generated
within the archive directory:

* `bentoml.yml` - a YAML file containing all metadata related to this
  BentoArchive
* `Dockerfile` - for building a Docker Image exposing this BentoService as REST
  API endpoint
* `setup.py` - the config file that makes a BentoArchive 'pip' installable

### Deployment & Inference Scenarios

- [Serving via REST API](#serving-via-rest-api)
- [Run REST API server with Docker](#run-rest-api-server-with-docker)
- [Loading BentoService in Python](#loading-bentoservice-in-python)
- [Use as PyPI Package](#use-as-pypi-package)
- [Use as CLI tool](#use-as-cli-tool)


#### Serving via REST API

For exposing your model as a HTTP API endpoint, you can simply use the `bentoml
serve` command:

```bash
bentoml serve ./bento_archive/IrisClassifier/v0.0.1/
```

Note you must ensure the pip and conda dependencies are available in your python
environment when using `bentoml serve` command. More commonly we recommend using
BentoML API server with Docker:

#### Run REST API server with Docker

You can build a Docker Image for running API server hosting your BentoML archive
by using the archive folder as docker build context:

```bash
cd ./bento_archive/IrisClassifier/v0.0.1/

docker build -t iris-classifier .
```

Next, you can `docker push` the image to your choice of registry for deployment,
or run it locally for development and testing:

```
docker run -p 5000:5000 iris-classifier
```

#### Loading BentoService in Python

`bentoml.load` is the enssential API for loading a BentoArchive into your
python application:

```python
import bentoml

# yes it works with BentoArchive saved to s3 ;)
bento_svc = bentoml.load('s3://my-bento-svc/iris_classifier/')
bento_svc.predict(X[0])
```

#### Use as PyPI Package

BentoML also supports distributing a BentoService as PyPI package, with the
generated `setup.py` file. A BentoArchive can be installed with `pip`:

```bash
pip install ./bento_archive/IrisClassifier/v0.0.1/
```

```python
import IrisClassifier

installed_svc = IrisClassifier.load()
installed_svc.predict(X[0])
```

With the `setup.py` config, a BentoArchive can also be uploaded to pypi.org
as a public python package, or to your organization's private PyPI index for all
developers in your organization to use:

```bash
cd ./bento_archive/IrisClassifier/v0.0.1/

# You will need a ".pypirc" config file before doing this:
# https://docs.python.org/2/distutils/packageindex.html
python setup.py sdist upload
```

#### Use as CLI tool

When `pip install` a BentoML archive, it also provides you with a CLI tool for
accessing your BentoService's APIs from the command line:

```bash
pip install ./bento_archive/IrisClassifier/v0.0.1/

IrisClassifier info  # this will also print out all APIs available

IrisClassifier predict --input='./test.csv'
```

Alternatively, you can also use the `bentoml` cli to load and run a BentoArchive
directly:

```bash
bentoml info ./bento_archive/IrisClassifier/v0.0.1/

bentoml predict ./bento_archive/IrisClassifier/v0.0.1/ --input='./test.csv'
```

### More About BentoML

We build BentoML because we think there should be a much simpler way for machine
learning teams to ship models for production. They should not wait for
engineering teams to re-implement their models for production environment or
build complex feature pipelines for experimental models.

Our vision is to empower Machine Learning scientists to build and ship their own
models end-to-end as production services, just like software engineers do.
BentoML is essentially this missing 'build tool' for Machine Learning projects.


## Examples

All examples can be found in the
[BentoML/examples](https://github.com/bentoml/BentoML/tree/master/examples)
directory.

- [Scikit-learn Sentiment Analysis](https://github.com/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb) - [![Google Colab Badge](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb)
- [H2O Classification](https://github.com/bentoml/BentoML/blob/master/examples/h2o-classification/h2o-example.ipynb) - [![Google Colab Badge](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/tf-keras-fashion-mnist/h2o-classification/h2o-example.ipynb)
- [Keras Text Classification](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-text-classification/tf-keras-text-classification.ipynb) - [![Google Colab Badge](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/tf-keras-text-classification/tf-keras-text-classification.ipynb)
- [XGBoost Titanic Survival Prediction](https://github.com/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb) - [![Google Colab Badge](https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal)](https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb)
- [(WIP) PyTorch Fashion MNIST classification](https://github.com/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb)
- [(WIP) Tensorflow Keras Fashion MNIST classification](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-fashion-mnist/tf-keras-fashion-mnist-classification.ipynb)


Deployment guides:
- [Serverless deployment with AWS Lambda](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-serverless)
- [API server deployment with AWS SageMaker](https://github.com/bentoml/BentoML/blob/master/examples/deploy-with-sagemaker)
- [API server deployment on Kubernetes](https://github.com/bentoml/BentoML/tree/master/examples/deploy-with-kubernetes)
- [API server deployment with Clipper (coming soon)](https://github.com/bentoml/BentoML/pull/151)


More tutorials and examples coming soon!


## Releases and Contributing

BentoML is under active development and is evolving rapidly. **Currently it is a
Beta release, we may change APIs in future releases**.

Want to help build BentoML? Check out our
[contributing documentation](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md).

To make sure you have a pleasant experience, please read the [code of conduct](https://github.com/bentoml/BentoML/blob/master/CODE_OF_CONDUCT.md).
It outlines core values and beliefs and will make working together a happier experience.



## License

BentoML is under Apache License 2.0, as found in the LICENSE file.


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fbentoml%2FBentoML.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fbentoml%2FBentoML?ref=badge_large) [![Join the chat at https://gitter.im/bentoml/BentoML](https://badges.gitter.im/bentoml/BentoML.svg)](https://gitter.im/bentoml/BentoML?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
