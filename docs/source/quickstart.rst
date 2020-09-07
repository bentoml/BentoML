.. _getting-started-page:

.. image:: https://static.scarf.sh/a.png?x-pxid=0beb35eb-7742-4dfb-b183-2228e8caf04c

Getting Started
###############

Run on Google Colab
^^^^^^^^^^^^^^^^^^^

Try out this quickstart guide interactively on Google Colab:
`Open in Colab <https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`_.

Note that Docker conterization does not work in the Colab environment.

Run notebook locally
^^^^^^^^^^^^^^^^^^^^

Install `BentoML <https://github.com/bentoml/BentoML>`_. This requires python 3.6 or
above, install with :code:`pip` command:

.. code-block:: bash

    pip install bentoml


Download and run the notebook in this quickstart guide:

.. code-block:: bash

    # Download BentoML git repo
    git clone http://github.com/bentoml/bentoml
    cd bentoml

    # Install jupyter and other dependencies
    pip install jupyter
    pip install ./guides/quick-start/requirements.txt

    # Run the notebook
    jupyter notebook ./guides/quick-start/bentoml-quick-start-guide.ipynb


Alternatively, :download:`Download the notebook <https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`
(Right-Click and then "Save Link As") to your notebook workspace.

To build a model server docker image, you will also need to install
:code:`docker` for your system, read more about how to install docker
`here <https://docs.docker.com/install/>`_.


Hello World
-----------

Before starting, let's prepare a trained model for serving with BentoML. Train a
classifier model with Scikit-Learn on the
`Iris data set <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_:

.. code-block:: python

    from sklearn import svm
    from sklearn import datasets

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)


Model serving with BentoML comes after a model is trained. The first step is creating a
prediction service class, which defines the models required and the inference APIs which
contains the serving logic. Here is a minimal prediction service created for serving
the iris classifier model trained above:

.. code-block:: python

    import pandas as pd

    from bentoml import env, artifacts, api, BentoService
    from bentoml.adapters import DataframeInput
    from bentoml.artifact import SklearnModelArtifact

    @env(auto_pip_dependencies=True)
    @artifacts([SklearnModelArtifact('model')])
    class IrisClassifier(BentoService):

        @api(input=DataframeInput())
        def predict(self, df: pd.DataFrame):
            # Optional pre-processing, post-processing code goes here
            return self.artifacts.model.predict(df)


The :code:`@api` decorator defines an inference API, which is the entry point for
accessing the prediction service. The :code:`input=DataframeInput()` means this inferene
API callback function defined by the user, is expecting a :code:`pandas.DataFrame`
object as its input.

In BentoML, all inference APIs are suppose to accept a list of inputs and return a list
of results. In the case of `DataframeInput`, each row of the dataframe is mapping to one
prediction request received from the client. BentoML will convert HTTP JSON requests
into :code:`pandas.DataFrame` object before passing it to the user-defined inference API
function.

This design allows BentoML to group API requests into small batches while serving online
traffic. Comparing to a regular flask or FastAPI based model server, this can increases
the overall throughput of the API server by 10-100x depending on the workload.

Besides `DataframeInput`, BentoML also supports API input types such as `JsonInput`,
`ImageInput`, `FileInput` and
`more <https://docs.bentoml.org/en/latest/api/adapters.html>`_.

The :code:`@env` decorator specifies the dependencies and environment settings
for this prediction service. It allows BentoML to reproduce the exact same environment
when moving the model and related code to production. With the
:code:`auto_pip_dependencies=True` flag used in this example, BentoML will automatically
figure all the PyPI packages that are required by the prediction service code and pins
down their versions.

Lastly the :code:`@artifact` defines the required trained models to be packed with this
prediction service. BentoML model artifact are pre-built wrappers for persisting a
traind model and access models from inference API. This example uses the
:code:`SklearnModelArtifact` for the sklearn model. BentoML also provide artifact class
for other frameworks including :code:`PytorchModelArtifact`, :code:`KerasModelArtifact`,
and :code:`XgboostModelArtifact` etc.


Save prediction service for distribution
----------------------------------------

The following code packages the trained model with the prediction service class
:code:`IrisClassifier` defined above, and then saves the IrisClassifier instance to disk
in the BentoML format for distribution and deployment:

.. code-block:: python

    # import the IrisClassifier class defined above
    from iris_classifier import IrisClassifier

    # Create a iris classifier service instance
    iris_classifier_service = IrisClassifier()

    # Pack the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to disk for model serving
    saved_path = iris_classifier_service.save()


BentoML stores all packaged model files under the
`~/bentoml/{service_name}/{service_version}` directory by default. The BentoML file
format contains all the code, files, and configs required to deploy the model for
serving.

BentoML also comes with a model management component called
`YataiService <https://docs.bentoml.org/en/latest/concepts.html#customizing-model-repository>`_,
which provides a central hub for teams to manage and access packaged models via Web UI
and API:

.. image:: _static/img/yatai-service-web-ui-repository.png
    :alt: BentoML YataiService Bento Repository Page

.. image:: _static/img/yatai-service-web-ui-repository-detail.png
    :alt: BentoML YataiService Bento Details Page


Launch Yatai server locally with docker and view your local repository of BentoML
packaged models:


.. code-block:: bash

    docker run -v ~/bentoml:/root/bentoml \
        -p 3000:3000 -p 50051:50051 \
        bentoml/yatai-service


.. note::

    The :code:`{saved_path}` in the following commands are referring to the returned
    value of :code:`iris_classifier_service.save()`.
    It is the file path where the BentoService saved bundle is stored.
    BentoML locally keeps track of all the BentoService SavedBundle you've created,
    you can also find the saved_path of your BentoService from the output of
    :code:`bentoml list -o wide`, :code:`bentoml get IrisClassifier -o wide` and
    :code:`bentoml get IrisClassifier:latest` command.

    A quick way of getting the :code:`saved_path` from the command line is via the
    `--print-location` option:

    .. code-block:: bash

        saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)



Model Serving via REST API
--------------------------

To start a REST API model server with the IrisClassifier saved above, use the
`bentoml serve` command followed by service name and version tag:

.. code-block:: bash

    bentoml serve IrisClassifier:latest

Alternatively, use the saved path to load and serve the BentoML packaged model directly:
.. code-block:: bash

    # Find the local path of the latest version IrisClassifier saved bundle
    saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

    bentoml serve $saved_path

The `IrisClassifier` model is now served at `localhost:5000`. Use `curl` command to send
a prediction request:

.. code-block:: bash

  curl -i \
    --header "Content-Type: application/json" \
    --request POST \
    --data '[[5.1, 3.5, 1.4, 0.2]]' \
    http://localhost:5000/predict

Or with :code:`python` and
`request library <https://requests.readthedocs.io/en/master/>`_:

.. code-block:: python

    import requests
    response = requests.post("http://127.0.0.1:5000/predict", json=[[5.1, 3.5, 1.4, 0.2]])
    print(response.text)


Note that BentoML API server automatically converts the Dataframe JSON format into a
`pandas.DataFrame` object before sending it to the user-defined inference API function.

The BentoML API server also provides a simple web UI dashboard.
Go to http://localhost:5000 in the browser and use the Web UI to send
prediction request:

.. image:: https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bento-api-server-web-ui.png
  :width: 600
  :alt: BentoML API Server Web UI Screenshot


Launch inference job from CLI
-----------------------------

BentoML cli supports loading and running a packaged model from CLI. With the
`DataframeInput` adapter, the CLI command supports reading input Dataframe data from CLI
argument or local csv or json files:

.. code-block:: bash

  bentoml run IrisClassifier:latest predict --input='[[5.1, 3.5, 1.4, 0.2]]'

  bentoml run IrisClassifier:latest predict --input='./iris_data.csv'


Containerize Model API Server
-----------------------------

One common way of distributing this model API server for production deployment, is via
Docker containers. And BentoML provides a convenient way to do that.

If you already have docker configured, run the following command to build a docker
container image for serving the `IrisClassifier` prediction service created above:


.. code-block:: bash

    bentoml containerize IrisClassifier:latest -t iris-classifier

Start a container with the docker image built from the previous step:

.. code-block:: bash

    docker run -p 5000:5000 iris-classifier:latest --workers=1 --enable-microbatch


If you need fine-grained control over how the docker image is built, BentoML provides a
convenient way to containerize the model API server manually:

.. code-block:: bash

    # 1. Find the SavedBundle directory with `bentoml get` command
    saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

    # 2. Run `docker build` with the SavedBundle directory which contains a generated Dockerfile
    docker build -t iris-classifier $saved_path

    # 3. Run the generated docker image to start a docker container serving the model
    docker run -p 5000:5000 iris-classifier --enable-microbatch --workers=1


This made it possible to deploy BentoML bundled ML models with platforms such as
`Kubeflow <https://www.kubeflow.org/docs/components/serving/bentoml/>`_,
`Knative <https://knative.dev/community/samples/serving/machinelearning-python-bentoml/>`_,
`Kubernetes <https://docs.bentoml.org/en/latest/deployment/kubernetes.html>`_, which
provides advanced model deployment features such as auto-scaling, A/B testing,
scale-to-zero, canary rollout and multi-armed bandit.

.. note::

  Ensure :code:`docker` is installed before running the command above.
  Instructions on installing docker: https://docs.docker.com/install


Deployment Options
------------------

If you are at a small team with limited engineering or DevOps resources, try out automated deployment with BentoML CLI, currently supporting AWS Lambda, AWS SageMaker, and Azure Functions:

- `AWS Lambda Deployment Guide <https://docs.bentoml.org/en/latest/deployment/aws_lambda.html>`_
- `AWS SageMaker Deployment Guide <https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html>`_
- `Azure Functions Deployment Guide <https://docs.bentoml.org/en/latest/deployment/azure_functions.html>`_

If the cloud platform you are working with is not on the list above, try out these step-by-step guide on manually deploying BentoML packaged model to cloud platforms:

- `AWS ECS Deployment <https://docs.bentoml.org/en/latest/deployment/aws_ecs.html>`_
- `Google Cloud Run Deployment <https://docs.bentoml.org/en/latest/deployment/google_cloud_run.html>`_
- `Azure container instance Deployment <https://docs.bentoml.org/en/latest/deployment/azure_container_instance.html>`_
- `Heroku Deployment <https://docs.bentoml.org/en/latest/deployment/heroku.html>`_

Lastly, if you have a DevOps or ML Engineering team who's operating a Kubernetes or OpenShift cluster, use the following guides as references for implementating your deployment strategy:

- `Kubernetes Deployment <https://docs.bentoml.org/en/latest/deployment/kubernetes.html>`_
- `Knative Deployment <https://docs.bentoml.org/en/latest/deployment/knative.html>`_
- `Kubeflow Deployment <https://docs.bentoml.org/en/latest/deployment/kubeflow.html>`_
- `KFServing Deployment <https://docs.bentoml.org/en/latest/deployment/kfserving.html>`_
- `Clipper.ai Deployment Guide <https://docs.bentoml.org/en/latest/deployment/clipper.html>`_


Distribute BentoML packaged model as a PyPI library
---------------------------------------------------

The BentoService SavedBundle is pip-installable and can be directly distributed as a
PyPI package if you plan to use the model in your python applications. You can install
it as as a system-wide python package with :code:`pip`:

.. code-block:: bash

  saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

  pip install $saved_path


.. code-block:: python

  # Your bentoML model class name will become the package name
  import IrisClassifier

  installed_svc = IrisClassifier.load()
  installed_svc.predict([[5.1, 3.5, 1.4, 0.2]])

This also allow users to upload their BentoService to pypi.org as public python package
or to their organization's private PyPi index to share with other developers.

.. code-block:: bash

    cd $saved_path & python setup.py sdist upload

.. note::

    You will have to configure ".pypirc" file before uploading to pypi index.
    You can find more information about distributing python package at:
    https://docs.python.org/3.7/distributing/index.html#distributing-index


Learning more about BentoML
---------------------------

Interested in learning more about BentoML? Check out the
:ref:`BentoML Core Concepts and best practices walkthrough <core-concepts-page>`,
a must-read for anyone who is looking to adopt BentoML.

Be sure to `join BentoML slack channel <http://bit.ly/2N5IpbB>`_ to hear about the
latest development updates and be part of the roadmap discussions.
