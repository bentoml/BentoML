# BentoML
> From a model in ipython notebook to production API service in 5 minutes.

![project status](https://www.repostatus.org/badges/latest/active.svg)
![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)
![pypi status](https://img.shields.io/pypi/v/bentoml.svg)
![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)


BentoML is a python library for packaging and deploying machine learning
models. It provides high-level APIs for defining a ML service and bundling
its artifacts, source code, dependencies, and configurations into a
production-system-friendly format that are ready for deployment.


---

- [Feature Highlights](#feature-highlights)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [More About BentoML](#more-about-bentoml)
- [Releases and Contributing](#releases-and-contributing)
- [License](#license)


## Feature Highlights

* __Multiple Distribution Format__ - Easily package your Machine Learning models
  into format that works best with your inference scenario:
  * Docker Image - include built-in REST API Server
  * PyPI Package - integrate with your python applications seamlessly
  * CLI tool - put your model into Airflow DAG or CI/CD pipeline
  * Spark UDF - run batch serving on large dataset with Spark
  * Serverless Function - host your model with serverless cloud platforms

* __Multiple Framework Support__ - BentoML supports a wide range of ML frameworks
  out-of-the-box including [Tensorflow](https://github.com/tensorflow/tensorflow/),
  [PyTorch](https://github.com/pytorch/pytorch),
  [Scikit-Learn](https://github.com/scikit-learn/scikit-learn),
  [xgboost](https://github.com/dmlc/xgboost) and can be easily extended to work
  with new or custom frameworks.

* __Deploy Anywhere__ - BentoML bundled ML service can be easily deploy with platforms
  such as [Docker](https://www.docker.com/), [Kubernetes](https://kubernetes.io/),
  [Serverless](https://github.com/serverless/serverless),
  [Airflow](https://airflow.apache.org) and [Clipper](http://clipper.ai),
  on cloud platforms including AWS Lambda/ECS/SageMaker, Gogole Cloud Functions, and
  Azure ML.

* __Custom Runtime Backend__ - Easily integrate your python preprocessing code with
  high-performance deep learning model runtime backend (such as
  [tensorflow-serving](https://github.com/tensorflow/serving)) to deploy low-latancy
  serving endpoint. 


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

Let's get started with a simple scikit-learn model as example:

```python
from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
```

To package this model with BentoML, you don't need to change anything in your
training code. Simply create a new BentoService by subclassing it:

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
packaging this BentoService. Other than `PickleArtifact`, bentoml also provides
`TfKerasModelArtifact`, `PytorchModelArtifact`, and `TfSavedModelArtifact` etc.

`@env` is designed for specifying the desired system environment in order for this
BentoService to load. Other ways you can use this decorator:

* If you already have a requirement.txt file listing all python libraries you
need:
```python
@env(requirement_txt='../myproject/requirement_txt')
```

* Or if you are running this code within a Conda environment that matches the
desired production environment:
```python
@env(with_current_conda_env=True)
```

Lastly `@api` adds an entry point for accessing this BentoService,
which will be translated into a REST endpoint when [deploying as API
server](#serving-via-rest-api), or a CLI command when [running as a CLI
tool](#use-as-cli-tool).

The `DataframeHandler` will transform either HTTP request or CLI command
arguments into a pandas Dataframe and pass it down to user defined API function.
BentoML also provides `JsonHandler`, 'ImageHandler' and `TensorHandler`.

Next, to save your trained model for prodcution use with this custom
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
environment when using `bentoml serve` command. More commonly we recommand using
BentoML API server with Docker:

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
accsing your BentoService's APIs from command line:

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


## Examples

All examples can be found in the
[BentoML/examples](https://github.com/bentoml/BentoML/tree/master/examples)
directory.

- [Quick Start with sklearn](https://github.com/bentoml/BentoML/blob/master/examples/quick-start/main.py)
- [Sentiment Analysis with Scikit-Learn](https://github.com/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb)
- [Text Classification with Tensorflow Keras](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-text-classification/tf-keras-text-classification.ipynb)
- [Fashion MNIST classification with Pytorch](https://github.com/bentoml/BentoML/blob/master/examples/pytorch-fashion-mnist/pytorch-fashion-mnist.ipynb)
- [Fashion MNIST classification with Tensorflow Keras](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-fashion-mnist/tf-keras-fashion-mnist-classification.ipynb)
- More examples coming soon!


## More About BentoML

We build BentoML because we think there should be a much simpler way for machine
learning teams to ship models for production. They should not wait for
engineering teams to re-implement their models for production environment or
build complex feature pipelines for experimental models.

Our vision is to empower Machine Learning scientists to build and ship their own
models end-to-end as production services, just like software engineers do.
BentoML is enssentially this missing 'build tool' for Machine Learing projects.


## Releases and Contributing

BentoML is under active development. Current version is a beta release, **we may
change APIs in future releases**.

Want to help build BentoML? Check out our
[contributing documentation](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md).



## License

BentoML is GPL-3.0 licensed, as found in the COPYING file.

