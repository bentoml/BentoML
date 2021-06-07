.. _getting-started-page:

***************
Getting Started
***************


Run on Google Colab
-------------------

Try out this quickstart guide interactively on Google Colab:
`Open in Colab <https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`_.

Note that Docker containerization does not work in the Colab environment.

Run Notebook Locally
--------------------

Install `BentoML <https://github.com/bentoml/BentoML>`_. This requires python 3.6 or
above, install with :code:`pip` command:

.. code-block:: bash

    pip install bentoml

When referring the latest documentation instead of the stable release doc, it is
required to install the preview release of BentoML:

.. code-block:: bash

    pip install --pre -U bentoml


Download and run the notebook in this quickstart guide:

.. code-block:: bash

    # Download BentoML git repo
    git clone http://github.com/bentoml/bentoml
    cd bentoml

    # Install jupyter and other dependencies
    pip install jupyter
    pip install -r ./guides/quick-start/requirements.txt

    # Run the notebook
    jupyter notebook ./guides/quick-start/bentoml-quick-start-guide.ipynb


Alternatively, :download:`Download the notebook <https://raw.githubusercontent.com/bentoml/BentoML/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`
(Right-Click and then "Save Link As") to your notebook workspace.

To build a model server docker image, you will also need to install
:code:`docker` for your system, read more about how to install docker
`here <https://docs.docker.com/install/>`_.


Preface
-------

Before started, let's discuss how BentoML's project structure would look like. For most use-cases, users can follow this minimal scaffold
for deploying with BentoML to avoid any potential errors (example project structure can be found under `guides/quick-start <https://github.com/bentoml/BentoML/tree/master/guides/quick-start>`_):

.. code-block:: bash

    bento_deploy/
    ├── bento_packer.py       # responsible for packing BentoService
    ├── bento_service.py      # BentoService definition
    ├── model.py               # DL Model definitions
    ├── train.py               # training scripts
    └── requirements.txt


We then need to prepare a trained model before serving with BentoML. Train a
classifier model with Scikit-Learn on the
`Iris data set <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_:

.. code-block:: python

    # train.py
    from sklearn import svm
    from sklearn import datasets

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

Example: Hello World
--------------------

Model serving with BentoML comes after a model is trained. The first step is creating a
prediction service class, which defines the models required and the inference APIs which
contains the serving logic code. Here is a minimal prediction service created for
serving the iris classifier model trained above, which is saved under *bento_service.py*:

.. code-block:: python

    # bento_service.py
    import pandas as pd

    from bentoml import env, artifacts, api, BentoService
    from bentoml.adapters import DataframeInput
    from bentoml.frameworks.sklearn import SklearnModelArtifact

    @env(infer_pip_packages=True)
    @artifacts([SklearnModelArtifact('model')])
    class IrisClassifier(BentoService):
        """
        A minimum prediction service exposing a Scikit-learn model
        """

        @api(input=DataframeInput(), batch=True)
        def predict(self, df: pd.DataFrame):
            """
            An inference API named `predict` with Dataframe input adapter, which codifies
            how HTTP requests or CSV files are converted to a pandas Dataframe object as the
            inference API function input
            """
            return self.artifacts.model.predict(df)


Firstly, the :code:`@artifact(...)` here defines the required trained models to be
packed with this prediction service. BentoML model artifacts are pre-built wrappers for
persisting, loading and running a trained model. This example uses the
:code:`SklearnModelArtifact` for the scikit-learn framework. BentoML also provide
artifact class for other ML frameworks, including :code:`PytorchModelArtifact`,
:code:`KerasModelArtifact`, and :code:`XgboostModelArtifact` etc.

The :code:`@env` decorator specifies the dependencies and environment settings required
for this prediction service. It allows BentoML to reproduce the exact same environment
when moving the model and related code to production. With the
:code:`infer_pip_packages=True` flag, BentoML will automatically find all the PyPI
packages that are used by the prediction service code and pins their versions.

The :code:`@api` decorator defines an inference API, which is the entry point for
accessing the prediction service. The :code:`input=DataframeInput()` means this inference
API callback function defined by the user, is expecting a :code:`pandas.DataFrame`
object as its input.

When the `batch` flag is set to True, an inference APIs is suppose to accept a list of
inputs and return a list of results. In the case of `DataframeInput`, each row of the
dataframe is mapping to one prediction request received from the client. BentoML will
convert HTTP JSON requests into :code:`pandas.DataFrame` object before passing it to the
user-defined inference API function.

This design allows BentoML to group API requests into small batches while serving online
traffic. Comparing to a regular flask or FastAPI based model server, this can largely
increase the overall throughput of the API server.

Besides `DataframeInput`, BentoML also supports API input types such as `JsonInput`,
`ImageInput`, `FileInput` and
`more <https://docs.bentoml.org/en/latest/api/adapters.html>`_. `DataframeInput` and
`TfTensorInput` only support inference API with `batch=True`, while other input adapters
support either batch or single-item API.


Save prediction service for distribution
----------------------------------------

The following code packages the trained model with the prediction service class
:code:`IrisClassifier` defined above, and then saves the IrisClassifier instance to disk
in the BentoML format for distribution and deployment, under *bento_packer.py*:

.. code-block:: python

    # bento_packer.py

    # import the IrisClassifier class defined above
    from iris_classifier import IrisClassifier

    # Create a iris classifier service instance
    iris_classifier_service = IrisClassifier()

    # Pack the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to disk for model serving
    saved_path = iris_classifier_service.save()


BentoML stores all packaged model files under the
`~/bentoml/repository/{service_name}/{service_version}` directory by default. The
BentoML packaged model format contains all the code, files, and configs required to
run and deploy the model.

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

    docker run \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v ~/bentoml:/bentoml \
      -p 3000:3000 \
      -p 50051:50051 \
      bentoml/yatai-service:latest

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

To start a REST API model server locally with the IrisClassifier saved above, use the
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

The BentoML CLI supports loading and running a packaged model from CLI. With the `DataframeInput` adapter, the CLI command supports reading input Dataframe data directly from CLI arguments and local files:

.. code-block:: bash

  bentoml run IrisClassifier:latest predict --input '[[5.1, 3.5, 1.4, 0.2]]'

  bentoml run IrisClassifier:latest predict --input-file './iris_data.csv'

More details on running packaged models that use other input adapters here: `Offline Batch Serving <https://docs.bentoml.org/en/latest/guides/batch_serving.html>`_

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

    docker run -p 5000:5000 iris-classifier:latest --workers=2


If you need fine-grained control over how the docker image is built, BentoML provides a
convenient way to containerize the model API server manually:

.. code-block:: bash

    # 1. Find the SavedBundle directory with `bentoml get` command
    saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

    # 2. Run `docker build` with the SavedBundle directory which contains a generated Dockerfile
    docker build -t iris-classifier $saved_path

    # 3. Run the generated docker image to start a docker container serving the model
    docker run -p 5000:5000 iris-classifier --workers=2


This made it possible to deploy BentoML bundled ML models with platforms such as
`Kubeflow <https://www.kubeflow.org/docs/components/serving/bentoml/>`_,
`Knative <https://knative.dev/community/samples/serving/machinelearning-python-bentoml/>`_,
`Kubernetes <https://docs.bentoml.org/en/latest/deployment/kubernetes.html>`_, which
provides advanced model deployment features such as auto-scaling, A/B testing,
scale-to-zero, canary rollout and multi-armed bandit.

.. note::

  Ensure :code:`docker` is installed before running the command above.
  Instructions on installing docker: https://docs.docker.com/install


Other deployment options are documented in the
:ref:`BentoML Deployment Guide <deployments-page>`, including Kubernetes, AWS, Azure,
Google Cloud, Heroku, and etc.


Learning more about BentoML
---------------------------

Interested in learning more about BentoML? Check out the
:ref:`BentoML Core Concepts and best practices walkthrough <core-concepts-page>`,
a must-read for anyone who is looking to adopt BentoML.

Be sure to `join BentoML slack channel <http://bit.ly/2N5IpbB>`_ to hear about the
latest development updates and be part of the roadmap discussions.


.. spelling::

    pypirc
    pre
    installable