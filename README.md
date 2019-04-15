# BentoML
> From a model in ipython notebook to production ready API service in 5 minutes.

![project status](https://www.repostatus.org/badges/latest/active.svg)
![build status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)
![pypi status](https://img.shields.io/pypi/v/bentoml.svg)
![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)


BentoML is a python library for packaging and deploying machine learning models.
It does two things without changing your model training workflow:

* Standardize how to package your ML model for production, including its
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
from bentoml.handlers import DataframeHandler

@artifacts([PickleArtifact('model')])
@env(conda_dependencies=["scikit-learn"])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.model.predict(df)
```

Now, to save your trained model for prodcution use, simply import your
BentoService class and `pack` it with required artifacts:

```python
from iris_classifier import IrisClassifier

svc = IrisClassifier.pack(model=clf)

svc.save('./saved_bento', version='v0.0.1') # Saving archive to ./saved_bento/IrisClassifier/v0.0.1/
```

That's it. Now you have created your first BentoArchive. It's a directory
containing all the source code, data and configurations files required to run
this model in production. There are a few ways you could use this archive:


### Serving a BentoArchive via REST API

For exposing your model as a HTTP API endpoint, you can simply use the `bentoml
serve` command:

```bash
bentoml serve ./saved_bento/IrisClassifier/v0.0.1/
```

Note: you must ensure the pip and conda dependencies are available in your python
environment when using `bentoml serve` command. More commonly we recommand using
BentoML API server with Docker(see below).


### Build API server Docker Image from BentoArchive

You can build a Docker Image for running API server hosting your BentoML archive
by using the archive folder as docker build context:

```bash
cd ./saved_bento/IrisClassifier/v0.0.1/

docker build -t myorg/iris-classifier .
```

Next, you can `docker push` the image to your choice of registry for deployment,
or run it locally for development and testing:

```
docker run -p 5000:5000 myorg/iris-classifier
```

### Loading BentoArchive in Python

```python
import bentoml

bento_svc = bentoml.load('./saved_bento/IrisClassifier/v0.0.1/')
bento_svc.predict(X[0])
```

BentoML also supports loading an archive from s3 location directly:

```python
bento_svc = bentoml.load('s3://my-bento-svc/iris_classifier/')
```

### Install BentoArchive as PyPI package

First install your exported bentoml service with `pip`:

```bash
pip install ./saved_bento/IrisClassifier/v0.0.1/
```

Now you can import it and used it as a python module:
```python
from IrisClassifier import IrisClassifier

installed_svc = IrisClassifier()
installed_svc.predict(X[0])
```

Note that you could also publish your exported BentoService as a PyPI package as
a public python package on pypi.org or upload to your organization's private
PyPI index:

```bash
cd ./saved_bento/IrisClassifier/v0.0.1/

python setup.py sdist upload
```

### Loading BentoArchive from CLI

When `pip install` a BentoML archive, it also provides you with a CLI tool for
accsing your BentoService's apis from command line:
```bash
pip install ./saved_bento/IrisClassifier/v0.0.1/

IrisClassifier info

IrisClassifier predict --input='./test.csv'
```

Alternatively, you can also use the `bentoml` cli to load and run a BentoArchive
directly:

```bash
bentoml info ./saved_bento/IrisClassifier/v0.0.1/

bentoml predict ./saved_bento/IrisClassifier/v0.0.1/ --input='./test.csv'
```

CLI access made it very easy to put your saved BentoArchive into an Airflow
DAG, integrate your packaged ML model into testing environment or use it in
combination with other shell tools.


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

