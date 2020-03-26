Getting Started
===============

Installing BentoML
------------------

`BentoML <https://github.com/bentoml/BentoML>`_ requires python 3.6 or above, install
via :code:`pip`:

.. code-block:: bash

    pip install bentoml

Instructions for installing from source can be found in the
`development guide <https://github.com/bentoml/BentoML/blob/master/DEVELOPMENT.md>`_.


Download Quickstart Notebook
++++++++++++++++++++++++++++

Download and run the code in this quickstart locally:

.. code-block:: bash

    pip install jupyter
    git clone http://github.com/bentoml/bentoml
    jupyter notebook bentoml/guides/quick-start/bentoml-quick-start-guide.ipynb

In order to build model server docker image, you will also need to install
:code:`docker` for your system, read more about how to install docker
`here <https://docs.docker.com/install/>`_.


Alternatively, run the code in this guide here on Google's Colab:

.. image:: https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal
    :target: https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb
    :alt: Launch on Colab



Hello World
-----------

The first step of creating a prediction service with BentoML, is to write a prediction
service class inheriting from :code:`bentoml.BentoService`, and describe the required
model artifacts, environment dependencies and writing your service API call back
function. Here is what a simple prediction service looks like:

.. code-block:: python

  import bentoml
  from bentoml.handlers import DataframeHandler
  from bentoml.artifact import SklearnModelArtifact

  @bentoml.env(auto_pip_dependencies=True)
  @bentoml.artifacts([SklearnModelArtifact('model')])
  class IrisClassifier(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
          return self.artifacts.model.predict(df)


The :code:`bentoml.api` and :code:`DataframeHandler` here tells BentoML, that following
by it, is the service API callback function, and :code:`pandas.Dataframe` is its
expected input format.

The :code:`bentoml.env` decorator allows specifying the dependencies and environment
settings for this prediction service. Here we are using BentoML's
:code:`auto_pip_dependencies` fature which automatically extracts and bundles all pip
packages that are required for your prediction service and pins down their version.


Lastly :code:`bentoml.artifact` defines the required trained models to be
bundled with this prediction service. Here it is using the built-in
:code:`SklearnModelArtifact` and simply naming it 'model'. BentoML also provide model
artifact for other frameworks such as :code:`PytorchModelArtifact`,
:code:`KerasModelArtifact`, :code:`FastaiModelArtifact`, and
:code:`XgboostModelArtifact` etc.


From Model Training To Serving
------------------------------

Next, we train a classifier model with Iris dataset, and pack the trained model with an
instance of the :code:`IrisClassifier` BentoService defined above, and save the entire
prediction service.

.. code-block:: python

  from sklearn import svm
  from sklearn import datasets

  clf = svm.SVC(gamma='scale')
  iris = datasets.load_iris()
  X, y = iris.data, iris.target
  clf.fit(X, y)

  # Create a iris classifier service with the newly trained model
  iris_classifier_service = IrisClassifier()
  iris_classifier_service.pack("model", clf)

  # Save the entire prediction service to file bundle
  saved_path = iris_classifier_service.save()

With the :code:`BentoService#save` call, you've just created a BentoML SavedBundle. It
is a versioned file archive that is ready for model serving deployment. The file archive
directory contains the BentoService you defined, the trained model artifact, all the
local python code you imported and PyPI dependencies in a requirements.txt etc, all
bundled in one place.


.. note::

    The :code:`{saved_path}` in the following commands are referring to the returned
    value of :code:`iris_classifier_service.save()`.
    It is the file path where the BentoService saved bundle is stored.
    BentoML locally keeps track of all the BentoService SavedBundle you've created,
    you can also find the saved_path of your BentoService via
    :code:`bentoml list -o wide` or
    :code:`bentoml get IrisClassifier -o wide` command.


Model Serving via REST API
--------------------------

You can start a REST API server by specifying the BentoService's name and version, or
provide the file path to the saved bundle:

.. code-block:: bash

  bentoml serve IrisClassifier:latest
  # or
  bentoml serve {saved_path}

The REST API server provides web UI for testing and debugging the server. If you are
running this command on your local machine, visit http://127.0.0.1:5000 in your browser
and try out sending API request to the server.

.. image:: https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bento-api-server-web-ui.png
  :width: 600
  :alt: BentoML API Server Web UI Screenshot

You can also send prediction request with :code:`curl` from command line:

.. code-block:: bash

  curl -i \
    --header "Content-Type: application/json" \
    --request POST \
    --data '[[5.1, 3.5, 1.4, 0.2]]' \
    http://localhost:5000/predict

Or with :code:`python` and :code:`request` library:

.. code-block:: python

    import requests
    response = requests.post("http://127.0.0.1:5000/predict", json=[[5.1, 3.5, 1.4, 0.2]])
    print(response.text)



Batch Serving via CLI
---------------------

For batch offline serving or testing your prediction service on batch test data, you
can load the BentoService SavedBundle from command line and run the prediction task on
the given input dataset. e.g.:

.. code-block:: bash

  bentoml run IrisClassifier:latest predict --input='[[5.1, 3.5, 1.4, 0.2]]'

  bentoml run IrisClassifier:latest predict --input='./iris_test_data.csv'


Containerize Model API Server
-----------------------------

The BentoService SavedBundle directory is structured to work as a docker build context,
which can be used directly to build a API server docker container image:


.. code-block:: bash

  docker build -t my_api_server {saved_path}

  docker run -p 5000:5000 my_api_server


.. note::

  You will need to install :code:`docker` before running this.
  Follow instructions here: https://docs.docker.com/install


Deploy API server to the cloud
------------------------------

BentoML has a built-in deployment management tool called YataiService. YataiService can
be deployed separately to manage all your teams' trained models, BentoService bundles,
and active deployments in a central place. But you can also create standalone model
serving deployments with just the BentoML cli, which launches a local YataiService
backed by SQLite database on your machine.

BentoML has built-in support for deploying to multiple cloud platforms. For demo
purpose, let's now deploy the IrisClassifier service we just created, to
`AWS Lambda <https://aws.amazon.com/lambda/>`_ into a serverless API endpoint.

First you need to install the :code:`aws-sam-cli` package, which is required by BentoML
to work with AWS Lambda deployment:

.. code-block:: bash

    pip install -U aws-sam-cli==0.31.1


.. note::

    You will also need to configure your AWS account and credentials if you don't have
    it configured on your machine. You can do this either
    `via environment variables <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`_
    or through the :code:`aws configure` command: install `aws` cli command via
    :code:`pip install awscli` and follow
    `detailed instructions here <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration>`_.

Now you can run the :code:`bentoml deploy` command, to create a AWS Lambda deployment,
hosting the BentService you've created:


.. code-block:: bash

    # replace the version here with the generated version string when creating the BentoService SavedBundle
    bentoml lambda deploy quick-start-guide-deployment \
        -b=IrisClassifier:20191126125258_4AB1D4 \


Distribute BentoService as a PyPI package
-----------------------------------------

The BentoService SavedBundle is pip-installable and can be directly distributed as a
PyPI package if you plan to use the model in your python applications. You can install
it as as a system-wide python package with :code:`pip`:

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

    cd {saved_path} & python setup.py sdist upload

.. note::

    You will have to configure ".pypirc" file before uploading to pypi index.
    You can find more information about distributing python package at:
    https://docs.python.org/3.7/distributing/index.html#distributing-index

Interested in learning more about BentoML? Check out the
`Examples <https://github.com/bentoml/BentoML#examples>`_ on BentoML github repository.

Be sure to `join BentoML slack channel <http://bit.ly/2N5IpbB>`_ to hear about the latest
development updates.
