Quick Start
===========

Install BentoML
---------------
Install BentoML is straightforward.

.. code-block:: python

    pip install bentoml


Running the quick start project
-------------------------------

The easiest way to try out the quick start project is using Google's Colab to run the
quick start project.

.. image:: https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal
    :target: https://colab.research.google.com/github/bentoml/BentoML/blob/master/examples/quick-start/bentoml-quick-start-guide.ipynb
    :alt: Launch on Colab

You can also run the quick start project locally. Download the quick start
example by git clone the BentoML repo, and navigate to the quick start project
inside the `example` folder


.. code-block:: bash

    $ pip install jupyter
    $ git clone http://github.com/bentoml/bentoml
    $ cd bentoml
    $ jupyter notebook examples/quick-start/bentoml-quick-start-guide.ipynb


We will go through each cell inside the notebook with explanation follows
below.

Quick start walk through
------------------------

Add BentoML to the notebook and training classification model
*************************************************************
.. code-block:: python

    !pip install -I bentoml
    !pip install pandas sklearn

We use jupyter notebook's built-in magic command to download and install
python modules such as scikit-learn for our example model.
We also download and install BentoML for define ML service later on.

.. code-block:: python

    from sklearn import svm
    from sklearn import datasets

    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

We trained a classification model with scikit-learn's iris dataset.

Define machine learning service with BentoML
********************************************

.. code-block:: python
    :linenos:

    %%writefile iris_classifier.py
    from bentoml import BentoService, api, env, artifacts
    from bentoml.artifact import PickleArtifact
    from bentoml.handlers import DataframeHandler

    @artifacts([PickleArtifact('model')])
    @env(conda_pip_dependencies=["scikit-learn"])
    class IrisClassifier(BentoService):

        @api(DataframeHandler)
        def predict(self, df):
            return self.artifacts.model.predict(df)

**Line 1**: We use jupyter notebook's built-in magic command to save our ML
service definition into a python file

**Line 2**: We import BentoService, our ML service will build on top of this
by subclassing it. We also import decorators such as, artifacts, api and env
for defining our ML service.

* **artifacts** decorator define what artifacts are required for packaging
  this service.

* **env** decorator designed for specifying the desired system environment
  and dependencies in order for this service to load. For this project we
  are using conda environment. If you already have requirement.txt file
  listing all of the python libraries you need:

    ``@env(requirement_txt="../my_project/requirement.txt")``

* **api**: decorator allow us to add an entry point to accessing this service.
  Each *api* will be translated into a REST endpoint when deploying as API
  server, or a CLI command when running the service as CLI tool.


**Line 3**: Using **PickleArtifact** for packaging our classifier model. Beside
  PickleArtifact, BentoML offers `KerasModelArtifact`,
  `PytorchModelArtifact`, `H2oModelArtifact`, `XgboostModelArtifact` and etc.

**Line 4**: Each API endpoint requires a Handler for defining the expect input
  format. For this project, we are using **DataframeHandler** to transform
  either a HTTP request or CLI command argument into a pandas dataframe and
  pass it down to the user defined API function. BentoML also provides
  `JsonHandler`, `ImageHandler` and `TensorHandler`


**Line 6-7**: We defined what artifact need to be included for this service,
and giving it a name `model`, and include the  python library that we need
for this project.

**Line 8**: We created our ML service called IrisClassifier by subclassing
`BentoService`

**Line 10-12**: We defined a function called `predict`. It will return result
from the artifact, `model`, we defined earlier by calling `predict` on that
artifact. We expose this predict function as our api for the service with the
`api` decorator, and tell BentoML that the incoming data will be transformed
into pandas dataframe for the user defined `predict` function to consume.


Now we have defined the ML service with BentoML, we will package our trained
model next and save it as archive to the file system.

Save defined ML service as BentoML service archive
**************************************************

.. code-block:: python
    :linenos:

    from iris_classifier import IrisClassifier

    svc = IrisClassifier.pack(model=clf)
    saved_path = svc.save('/tmp/bentoml_archive')

**Line 1**: We import the service definition we wrote in the previous cell.

**Line 3**: We are packaging the trained model from above with the ML
service.

**Line 4-5**: We saved the packed service as BentoML archive into the local
file system and print out the saved location path.

We just created and saved our quick start project into BentoML service archive.
It is a directory containing all of the source code, data, and configurations
that required to load and run as Bento Service. You will find three `magic`
files that generated within the archive directory:

- bentoml.yml: A YAML file contains all of the metadata related to this service
  and archive.

- setup.py: The configuration file that makes this BentoML service archive
  'pip' installable

- Dockerfile: for building Docker image that expose this Bento service as REST
  API service.


Using BentoML archive
*********************

Real-time serving with REST API
+++++++++++++++++++++++++++++++
To exposing your ML service as HTTP API endpoint, you can simply use the
bentoml serve command:

.. code-block:: python

    !bentoml serve {saved_path}

With `bentoml serve` command, a web server will start locally at the port 5000.
We created additional endpoints that make this server ready for production.

- `/`: The index page with OpenAPI definition.

- `/docs.json`: The Open API definition for all endpoints in JSON format.

- `/metrics`: Expose system and latency metrics with Prometheus.

- `/healthz`: Check on your service health.

- `/feedback`: Add business feedback for the predicted results.

Open http://127.0.0.1:5000 to view the documentation for all API endpoints.

Run REST API server with Docker
+++++++++++++++++++++++++++++++
To deploy the Bento service as REST api server for production use, we can use
the generated Dockerfile to create Docker image for that.

.. code-block:: python

    !cd {saved_path} && docker build -t iris-classifier .

.. code-block:: python

    !docker run -p 5000:5000 iris-classifier


.. note::

    To generate Docker image, you will need to install Docker on your system. Please
    follow direction from this link: https://docs.docker.com/install


(Optional) Get a Client SDK for the above REST API server
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To get a client SDK, copy the content of http://127.0.0.1:5000/docs.json and paste to
https://editor.swagger.io then click the tab Generate Client and choose the language.

Currently, https://editor.swagger.io supports to generate a client SDK in Java, Kotlin,
Swift, Python, PHP, Scala... ect.

Loading Bento service archive in Python
+++++++++++++++++++++++++++++++++++++++

The easiest to use Bento service archive in your python application is using
`bentoml.load`.

.. code-block:: python

    import bentoml
    import pandas as pd

    bento_svc = bentoml.load(saved_path)
    bento_svc.predict([X[0]]


`pip install` a BentoML service archive
+++++++++++++++++++++++++++++++++++++++

BentoML support distributing Bento service as PyPi package, with the generated
`setup.py` file. Bento service archive can be installed with pip:

.. code-block:: python

    !pip install {saved_path}

Bento service archive can be uploaded to pypi.org as public python package or
to your organization's private PyPi index for all developers in your org to
use.

.. code-block:: bash

    !cd {saved_path} & python setup.py sdist upload

.. note::

    You will have to configure ".pypirc" file before uploading to pypi index.
    You can find more information about distributing python package at:
    https://docs.python.org/3.7/distributing/index.html#distributing-index


After pip install, we can import the Bento service as regular python package.

.. code-block:: python

    import IrisClassifier

    installed_svc = IrisClassifier.load()
    installed_svc.predict([X[0]]


CLI access with BentoML service archive
+++++++++++++++++++++++++++++++++++++++

`pip install` includes a CLI tool for accessing the Bento service.

From terminal, you can use `info` command to list all APIs defined in the
service.

.. code-block:: python

    !IrisClassifier info

You can use `docs` command to get all APIs in OpenAPI format.

.. code-block:: python

    !IrisClassifier docs

Call prediction with user defined API function.

.. code-block:: python

    !IrisClassifier predict --help

.. code-block:: python

    !IrisClassifier predict --input='[[5.1, 3.5, 1.4, 0.2]]'

Alternatively, use ``bentoml cli`` to load and run Bento service archive
without installing.

.. code-block:: python

    !bentoml info {saved_path}

.. code-block:: python

    !bentoml docs {saved_path}

.. code-block:: python

    !bentoml predict {saved_path} --input='[[5.1, 3.5, 1.4, 0.2]]'


Congratulation! You've train, build, and running your first Bento
service.
