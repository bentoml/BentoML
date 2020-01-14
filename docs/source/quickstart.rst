Quick Start
===========

Installing BentoML
------------------

BentoML requires python 3.6 or above, install via `pip`:

.. code-block:: bash

    $ pip install bentoml

Instructions for installing from source can be found in the
`development guide <https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md>`_.


Download Quickstart Notebook
----------------------------

Download and run the code in this quickstart locally:

.. code-block:: bash

    $ pip install jupyter
    $ git clone http://github.com/bentoml/bentoml
    $ jupyter notebook bentoml/guides/quick-start/bentoml-quick-start-guide.ipynb

In order to build model server docker image, you will also need to install `docker` for your system,
read more about how to install docker `here <https://docs.docker.com/install/>`_.


Alternatively, run the code in this guide here on Google's Colab:

.. image:: https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal
    :target: https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb
    :alt: Launch on Colab



Creating a Prediction Service with BentoML
------------------------------------------

The first step of creating a prediction service with BentoML, is to write a prediction
service class inheriting from `bentoml.BentoService`, and declaratively listing the
dependencies, model artifacts and writing your service API call back function. Here is
what a simple prediction service looks like:

.. code-block:: python

  import bentoml
  from bentoml.handlers import DataframeHandler
  from bentoml.artifact import SklearnModelArtifact

  @bentoml.env(pip_dependencies=["scikit-learn"])
  @bentoml.artifacts([SklearnModelArtifact('model')])
  class IrisClassifier(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
          return self.artifacts.model.predict(df)


The `bentoml.api` and `DataframeHandler` here tells BentoML, that following by it, is
the service API callback function, and `pandas.Dataframe` is its expected input format.

The `bentoml.env` decorator allows user to specify the dependencies and environment 
settings for this prediction service. Here we are creating the prediction service based
on a scikit learn model, so we add it to the list of pip dependencies.


Last but not least, `bentoml.artifact` declares the required trained model to be bundled
with this prediction service. Here it is using the built-in `SklearnModelArtifact` and
simply naming it 'model'. BentoML also provide model artifact for other frameworks such
as `PytorchModelArtifact`, `KerasModelArtifact`, `FastaiModelArtifact`, and
`XgboostModelArtifact` etc.


Saving a versioned BentoService bundle
--------------------------------------

Next, we train a classifier model with Iris dataset, and pack the trained model with the
BentoService `IrisClassifier` defined above:

.. code-block:: python

  from sklearn import svm
  from sklearn import datasets

  clf = svm.SVC(gamma='scale')
  iris = datasets.load_iris()
  X, y = iris.data, iris.target
  clf.fit(X, y)

  # Create a iris classifier service with the newly trained model
  iris_classifier_service = IrisClassifier.pack(model=clf)

  # Save the entire prediction service to file bundle
  saved_path = iris_classifier_service.save()


You've just created a BentoService SavedBundle, it's a versioned file archive that is
ready for production deployment. It contains the BentoService you defined, as well as
the packed trained model artifacts, pre-processing code, dependencies and other
configurations in a single file directory.


Model Serving via REST API
++++++++++++++++++++++++++

From a BentoService SavedBundle, you can start a REST API server by providing the file
path to the saved bundle:

.. code-block:: bash

  bentoml serve {saved_path}


The REST API server provides a simply web UI for you to test and debug. If you are
running this command on your local machine, visit http://127.0.0.1:5000 in your browser
and try out sending API request to the server.

.. image:: https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bento-api-server-web-ui.png
  :width: 600
  :alt: BentoML API Server Web UI Screenshot

You can also send prediction request with `curl` from command line:

.. code-block:: bash

  curl -i \
    --header "Content-Type: application/json" \
    --request POST \
    --data '[[5.1, 3.5, 1.4, 0.2]]' \
    http://localhost:5000/predict

Or with `python` and `request` library:

.. code-block:: python

    import requests
    response = requests.post("http://127.0.0.1:5000/predict", json=[[5.1, 3.5, 1.4, 0.2]])
    print(response.text)


Model Serving via CLI
+++++++++++++++++++++

For testing purpose, you can load the BentoService SavedBundle from command line and
run the prediction task on the given input dataset:

.. code-block:: bash

  bentoml predict {saved_path} --input='[[5.1, 3.5, 1.4, 0.2]]'

  # alternatively:
  bentoml predict {saved_path} --input='./iris_test_data.csv'



Distribute BentoML SavedBundle as PyPI package
+++++++++++++++++++++++++++++++++++++++++

The BentoService SavedBundle is pip-installable and can be directly distributed as a
PyPI package if you plan to use the model in your python applications. You can install
it as as a system-wide python package with `pip`:

.. code-block:: bash

  pip install {saved_path}

.. code-block:: python

  # Your bentoML model class name will become packaged name
  import IrisClassifier

  installed_svc = IrisClassifier.load()
  installed_svc.predict([[5.1, 3.5, 1.4, 0.2]])

This also allow users to upload their BentoService to pypi.org as public python package
or to their organization's private PyPi index to share with other developers.

.. code-block:: bash

    !cd {saved_path} & python setup.py sdist upload

.. note::

    You will have to configure ".pypirc" file before uploading to pypi index.
    You can find more information about distributing python package at:
    https://docs.python.org/3.7/distributing/index.html#distributing-index


Containerize REST API server with Docker
++++++++++++++++++++++++++++++++++++++++

The BentoService SavedBundle is structured to work as a docker build context, that can
be directed used to build a docker image for API server. Simply use it as the docker
build context directory:


.. code-block:: bash

  docker build -t my_api_server {saved_path}

  docker run -p 5000:5000 my_api_server


.. note::

  You will need to install Docker before running this.
  Follow direction from this link: https://docs.docker.com/install





Learning More?
++++++++++++++

Interested in learning more about BentoML? Check out the
`Examples <https://github.com/bentoml/BentoML#examples>`_ on BentoML github repository.

Be sure to `join BentoML slack channel <http://bit.ly/2N5IpbB>`_ to hear about the latest
development updates.
