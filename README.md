# BentoML
From a model in ipython notebook, to production ready API service in 5 minutes

![project status](https://www.repostatus.org/badges/latest/active.svg)
![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)
![pypi status](https://img.shields.io/pypi/v/bentoml.svg)
![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)


BentoML is a python library for packaging and deploying machine learning models.
It does two things without changing your model training workflow:

* Standarlize how to package your ML model for production, including its
  preprocessing/feature-fetching code, dependencies and configurations.

* Easily distribute your ML model as PyPI package, API Server(in a Docker Image)
  , command line tool or Spark/Flink UDF.

---

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [More About BentoML](#more-about-bentoml)
- [Releases and Contributing](#releases-and-contributing)
- [License](#license)


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

BentoML does not change your training workflow. Let's train a simple
scikit-learn model as example:

```python
from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
```

To package this model with BentoML, you will need to create a new BentoService
by subclassing it, and provides artifacts and env definition for it:

```python
%%writefile iris_classifier.py
from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler

@artifacts([PickleArtifact('model')])
@env(conda_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(JsonHandler)
    def predict(self, parsed_json):
        return self.artifacts.model.predict(parsed_json)
```

Now, to save your trained model for prodcution use, simply import your
BentoService class and `pack` it with required artifacts:

```
from iris_classifier import IrisClassifier

svc = IrisClassifier.pack(model=clf)

svc.save('./saved_bento', version='v0.0.1') # Saving archive to ./saved_bento/IrisClassifier/v0.0.1/
```

That's it. Now you have created your first BentoML archive. It's a directory
containing all the source code, data files and configurations required to run
this model in production. There are a few ways you could use this archive:


### Load BentoML archive in Python

```
import bentoml

bento_svc = bentoml.load('./saved_bento/IrisClassifier/v0.0.1/')
bento_svc.predict(X[0])
```

### Import as a python package

First install your exported model service with `pip`:
```bash
pip install ./saved_bento/IrisClassifier/v0.0.1/
```

Now you can import it and used it as a python module:
```python
from IrisClassifier import IrisClassifier

installed_svc = IrisClassifier.load()
installed_svc.predict(X[0])
```

Note that you could also publish your exported BentoService as a PyPI package as
a public python package on pypi.org or upload to your organization's private
PyPI index:

```bash
cd ./saved_bento/IrisClassifier/v0.0.1/

python setup.py sdist upload
```

### Run as a CLI tool

When `pip install` a BentoML archive, it also provides you with a CLI tool for
accsing your BentoService's apis from command line:
```bash
pip install ./saved_bento/IrisClassifier/v0.0.1/

IrisClassifier --help

IrisClassifier predict --input='[5.1, 3.5, 1.4, 0.2]'
```

### Run archive as a REST API server

For exposing your model as a HTTP API endpoint, you can simply use the `bentoml
serve` command:

```bash
bentoml serve --archive-path="./saved_bento/IrisClassifier/v0.0.1/"
```

### Build API server Docker Image

To make it easier for your DevOps colleagues, a image containing the API server
can be build directly using the archive folder as docker build context:

```bash
cd ./saved_bento/IrisClassifier/v0.0.1/

docker build -t myorg/iris-classifier .
```


## Examples

All examples can be found in the
[BentoML/examples](https://github.com/bentoml/BentoML/tree/master/examples)
directory.

- [Quick Start with sklearn](https://github.com/bentoml/BentoML/blob/master/examples/quick-start/main.py)
- [Sentiment Analysis with Scikit-Learn](https://github.com/bentoml/BentoML/blob/master/examples/sklearn-sentiment-clf/sklearn-sentiment-clf.ipynb)
- [Text Classification with Tensorflow Keras](https://github.com/bentoml/BentoML/blob/master/examples/tf-keras-text-classification/tf-keras-text-classification.ipynb)
- More examples coming soon!


## More About BentoML

We build BentoML because we think there should be a much simpler way for machine
learning teams to ship models for production. They should not wait for
engineering teams to re-implement their models for production environment or
build complex feature pipelines for experimental models.

Our vision is to empower Machine Learning scientists to build and ship their own
models as production services, just like software engineers do. BentoML is
enssentially this missing 'build tool' for Machine Learing projects.

With that in mind, here is the top design goals for BentoML:

* Multiple framework support - BentoML should supports a wide range of ML
frameworks out-of-the-box including Tensorflow, PyTorch, Scikit-Learn, xgboost
and can be easily extended to work with new or custom frameworks.

* Best Practice built-in - BentoML users can easily customize telemetrics and
logging for their model, and make it easy to integrate with production systems.

* Streamlines deployment workflows - BentoML supports deploying models into REST
API endpoints with Docker, Kubernetes, AWS EC2, ECS, Google Cloud Platform, AWS
SageMaker, and Azure ML.

* Custom model runtime - Easily integrate your python code with high-performance
model runtime backend(e.g. tf-serving, tensorrt-inference-server) in real-time
model serving.



## Releases and Contributing

BentoML is under active development. Current version is a beta release, **we may
change APIs in future releases**.

Want to help build BentoML? Check out our
[contributing documentation](https://github.com/bentoml/BentoML/blob/master/CONTRIBUTING.md).



## License

BentoML is GPL-3.0 licensed, as found in the COPYING file.

