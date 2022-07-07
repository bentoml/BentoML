==========================
Tutorial: Intro to BentoML
==========================

BentoML is a python-first, efficient and flexible framework for machine learning model
serving. It lets data scientists save and version trained models in a standardized
format and unifies how a saved model can be accessed for serving. This enables ML
engineers to easily use the saved models for building online prediction services or
batch inference jobs.

BentoML also helps with defining the APIs, environments, and dependencies for running a
model, providing a build tool that encapsulates all model artifacts, source code, and
dependencies into a self-contained format :code:`Bento`, which is designed to be DevOps
friendly and ready for production deployment - just like docker but designed for ML
models.

What are we building?
---------------------

In this tutorial, we will focus on online model serving with BentoML, using a
classification model trained with Scikit-Learn and the Iris dataset. By the end of this
tutorial, we will have an HTTP endpoint for handling inference requests and a docker
container image for deployment.


.. note::

    You might be tempted to skip this tutorial because you are not using scikit-learn,
    but give it a chance. The concepts you will learn in the tutorial are fundamental to
    model serving with any ML framework using BentoML, and mastering it will give you a
    deep understanding of BentoML.


Setup for the tutorial
----------------------

There are three ways to complete this tutorial:

#. Run with Google Colab in your browser

   👉 `Open Tutorial Notebook on Colab <https://colab.research.google.com/github/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb>`_
   side by side with this guide. As you go through this guide, you can simply run the
   sample code from the Colab Notebook.

   You will be able to try out most of the content in the tutorial on Colab besides
   the docker container part towards the end. This is because Google Colab currently
   does not support docker.

#. Run the tutorial notebook from Docker

   If you have Docker installed, you can run the tutorial notebook from a pre-configured
   docker image with:

   .. code:: bash

      docker run -it --rm -p 8888:8888 -p 3000:3000 bentoml/quickstart:latest

#. Local Development Environment

   Download the source code of this tutorial from `bentoml/Gallery <https://github.com/bentoml/gallery/>`_:

   .. code:: bash

      git clone --depth=1 git@github.com:bentoml/gallery.git
      cd gallery/quickstart/

   BentoML supports Linux, Windows and MacOS. You will need Python 3.7 or above to run
   this tutorial. We recommend using `virtual environment <https://docs.python.org/3/library/venv.html>`_
   to create an isolated local environment. However this is not required.

   Install all dependencies required for this tutorial:

   .. code-block:: bash

       pip install --pre bentoml
       pip install scikit-learn pandas


Saving Models with BentoML
--------------------------

To begin with BentoML, you will need to save your trained models with BentoML API in
its model store (a local directory managed by BentoML). The model store is used for
managing all your trained models locally as well as accessing them for serving.

.. code-block:: python
   :emphasize-lines: 14,15

   import bentoml

   from sklearn import svm
   from sklearn import datasets

   # Load training data set
   iris = datasets.load_iris()
   X, y = iris.data, iris.target

   # Train the model
   clf = svm.SVC(gamma='scale')
   clf.fit(X, y)

   # Save model to the BentoML local model store
   bentoml.sklearn.save_model("iris_clf", clf)
   print(f"Model saved: {saved_model}")

   # Model saved: Model(tag="iris_clf:zy3dfgxzqkjrlgxi")


The model is now saved under the name :code:`iris_clf` with an automatically generated
version. The name and version pair can then be used for retrieving the model. For
instance, the original model object can be loaded back into memory for testing via:

.. code-block::

   model = bentoml.sklearn.load_model("iris_clf:2uo5fkgxj27exuqj")

   # Alternatively, use `latest` to find the newest version
   model = bentoml.sklearn.load_model("iris_clf:latest")


The :code:`bentoml.sklearn.save_model` API is built specifically for the Scikit-Learn
framework and uses its native saved model format under the hood for best compatibility
and performance. This goes the same for other ML frameworks, e.g.
:code:`bentoml.pytorch.save_model`, see the :doc:`frameworks/index` to learn more.


.. seealso::

   It is possible to use pre-trained models directly with BentoML or import existing
   trained model files to BentoML. Learn more about it from :doc:`concepts/model`.


Saved models can be managed via the :code:`bentoml models` CLI command or Python API,
learn about it here: :ref:`concepts/model:Managing Models`.


Creating a Service
------------------

Services are the core components of BentoML, where the serving logic is defined. Create
a file :code:`service.py` with:

.. code:: python

    import numpy as np
    import bentoml
    from bentoml.io import NumpyNdarray

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result


Run it live:

.. code:: bash

    > bentoml serve service:svc --reload

    2022-07-01T14:19:12-0700 [INFO] [] Starting development BentoServer from "service:svc" running on http://127.0.0.1:3000 (Press CTRL+C to quit)
    2022-07-01 14:19:12 circus[10959] [INFO] Loading the plugin...
    2022-07-01 14:19:12 circus[10959] [INFO] Endpoint: 'tcp://127.0.0.1:52870'
    2022-07-01 14:19:12 circus[10959] [INFO] Pub/sub: 'tcp://127.0.0.1:52871'
    2022-07-01 14:19:12 asyncio[10959] [DEBUG] Using selector: KqueueSelector
    2022-07-01 14:19:12 asyncio[10959] [DEBUG] Using selector: KqueueSelector
    2022-07-01 14:19:12 bentoml._internal.utils.circus[10959] [INFO] Watching directories: /home/user/gallery/quickstart
    2022-07-01 14:19:12 circus[10959] [INFO] Starting

.. dropdown:: About the command :code:`bentoml serve service:svc --reload`
   :icon: code

   In the example above:

   - :code:`service` refers to the python module (the :code:`service.py` file)
   - :code:`svc` refers to the object created in :code:`service.py`, with :code:`svc = bentoml.Service(...)`
   - :code:`--reload` option watches for local code changes and automatically restart server. This is for development use only.

   .. tip::

      This syntax also applies to projects with nested directories. For example, if you
      have a :code:`./src/foo/bar/my_service.py` file where a service object is defined
      with: :code:`my_bento_service = bentoml.Service(...)`, the command will be:

      .. code:: bash

         bentoml serve src.foo.bar.my_service:my_bento_service
         # Or
         bentoml serve ./src/foo/bar/my_service.py:my_bento_service


Send prediction requests with an HTTP client:

.. tab-set::
   .. tab-item:: Python

      .. code:: python

         import requests
         requests.post(
             "http://127.0.0.1:3000/classify",
             headers={"content-type": "application/json"},
             data="[[5.9, 3, 5.1, 1.8]]").text

   .. tab-item:: Curl

      .. code:: bash

         curl \
           -X POST \
           -H "content-type: application/json" \
           --data "[[5.9, 3, 5.1, 1.8]]" \
           http://127.0.0.1:3000/classify

   .. tab-item:: Browser

      Open http://127.0.0.1:3000 in your browser and send test request from the web UI.


Using Models in a Service
~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, :code:`bentoml.sklearn.get` creates a reference to the saved model
in the model store, and :code:`to_runner` creates a Runner instance from the model.
The Runner abstraction gives BentoServer more flexibility in terms of how to schedule
the inference computation, how to dynamically batch inference calls and better take
advantage of all hardware resource available.

You can test out the Runner interface this way:

.. code:: python

   import bentoml

   iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
   iris_clf_runner.init_local()
   iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]])

.. note::

   For custom Runners and advanced runner options, see :doc:`concepts/runner` and :doc:`guides/batching`.


Service API and IO Descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`svc.api` decorator adds a function to the :code:`bentoml.Service` object's
APIs list. The :code:`input` and :code:`output` parameter takes an
:doc:`IO Descriptor <reference/api_io_descriptors>` object, which specifies the API
function's expected input/output types, and is used for generating HTTP endpoints.

In this example, both :code:`input` and :code:`output` are defined with
:ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy ndarray>`, which means
the API function being decorated, takes a :code:`numpy.ndarray` as input, and returns a
:code:`numpy.ndarray` as output.

.. note::

   More options, such as :code:`pandas.DataFrame`, :code:`Json`, and :code:`PIL.image`
   are also supported. An IO Descriptor object can also be configured with a schema or
   a shape for input/output validation. Learn more about them in
   :doc:`reference/api_io_descriptors`.

Inside the API function, users can define any business logic, feature fetching, and
feature transformation code. Model inference calls are made directly through runner
objects, that are passed into :code:`bentoml.Service(name=.., runners=[..])` call when
creating the service object.

.. tip::

   BentoML supports both :ref:`Sync and Async endpoints <concepts/service:Sync vs Async APIs>`.
   For performance sensitive use cases, especially when working with IO-intense
   workloads (e.g. fetching features from multiple sources) or when
   :doc:`composing multiple models <guides/multi_models>`, you may consider defining an
   :code:`Async` API instead.

   Here's an example of the same endpoint above defined with :code:`Async`:

   .. code:: python

      @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
      async def classify(input_series: np.ndarray) -> np.ndarray:
         result = await iris_clf_runner.predict.async_run(input_series)
         return result


Building a Bento 🍱
-------------------

Once the service definition is finalized, we can build the model and service into a
:code:`bento`. Bento is the distribution format for a service. It is a self-contained
archive that contains all the source code, model files and dependency specifications
required to run the service.

To build a Bento, first create a :code:`bentofile.yaml` file in your project directory:

.. code:: yaml

   service: "service:svc"  # Same as the argument passed to `bentoml serve`
   labels:
      owner: bentoml-team
      stage: dev
   include:
   - "*.py"  # A pattern for matching which files to include in the bento
   python:
      packages:  # Additional pip packages required by the service
      - scikit-learn
      - pandas

.. tip::

   BentoML provides lots of build options in :code:`bentofile.yaml` for customizing the
   Python dependencies, cuda installation, docker image distro, etc. Read more about it
   on the :doc:`concepts/bento` page.


Next, run the :code:`bentoml build` CLI command from the same directory:

.. code:: bash

    > bentoml build

    Building BentoML service "iris_classifier:6otbsmxzq6lwbgxi" from build context "/home/user/gallery/quickstart"
    Packing model "iris_clf:zy3dfgxzqkjrlgxi"
    Locking PyPI package versions..
 
    ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
    ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
    ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
    ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
    ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
    ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
 
    Successfully built Bento(tag="iris_classifier:6otbsmxzq6lwbgxi")

🎉 You've just created your first Bento, and it is now ready for serving in production!
For starters, you can now serve it with the :code:`bentoml serve` CLI command:

.. code:: bash

    > bentoml serve iris_classifier:latest --production

    2022-07-01T14:48:47-0700 [INFO] [] Starting production BentoServer from "iris_classifier:latest" running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    2022-07-01T14:48:49-0700 [INFO] [runner-iris_clf:1] Setting up worker: set CPU thread count to 10


.. note::

   Even though the service definition code uses model :code:`iris_clf:latest`, the
   :code:`latest` version can be resolved with local model store to find the exact model
   version :code:`demo_mnist:7drxqvwsu6zq5uqj` during the :code:`bentoml build`
   process. This model is then bundled into the Bento, which makes sure this Bento is
   always using this exact model version, wherever it is deployed.


Bento is the unit of deployment in BentoML, one of the most important artifacts to keep
track of in your model deployment workflow. BentoML provides CLI commands and APIs for
managing Bentos and moving them around, see the :ref:`concepts/bento:Managing Bentos`
section to learn more.


Generate Docker Image
---------------------

A docker image can be automatically generated from a Bento for production deployment,
via the :code:`bentoml containerize` CLI command:

.. code:: bash

   > bentoml containerize iris_classifier:latest

   Building docker image for Bento(tag="iris_classifier:6otbsmxzq6lwbgxi")...
   Successfully built docker image "iris_classifier:6otbsmxzq6lwbgxi"

.. note::

   You will need to `install Docker <https://docs.docker.com/get-docker/>`_ before
   running this command.

.. dropdown:: For Mac with Apple Silicon
   :icon: cpu

   Specify the :code:`--platform` to avoid potential compatibility issues with some
   Python libraries.

   .. code:: bash

      bentoml containerize --platform=linux/amd64 iris_classifier:latest

This creates a docker image that includes the Bento, and has all its dependencies
installed. The docker image tag will be same as the Bento tag by default:

.. code:: bash

   > docker images

   REPOSITORY         TAG                 IMAGE ID        CREATED          SIZE
   iris_classifier    6otbsmxzq6lwbgxi    0b4f5ec01941    10 seconds ago   790MB


Run the docker image to start the BentoServer:

.. code:: bash

    docker run -p 3000:3000 iris_classifier:6otbsmxzq6lwbgxi

    2022-07-01T21:57:47+0000 [INFO] [] Service loaded from Bento directory: bentoml.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/bentoml/bento/")
    2022-07-01T21:57:47+0000 [INFO] [] Starting production BentoServer from "/home/bentoml/bento" running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    2022-07-01T21:57:48+0000 [INFO] [api_server:1] Service loaded from Bento directory: bentoml.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/bentoml/bento/")
    2022-07-01T21:57:48+0000 [INFO] [runner-iris_clf:1] Service loaded from Bento directory: bentoml.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/bentoml/bento/")
    2022-07-01T21:57:48+0000 [INFO] [api_server:2] Service loaded from Bento directory: bentoml.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/bentoml/bento/")
    2022-07-01T21:57:48+0000 [INFO] [runner-iris_clf:1] Setting up worker: set CPU thread count to 4
    2022-07-01T21:57:48+0000 [INFO] [api_server:3] Service loaded from Bento directory: bentoml.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/bentoml/bento/")
    2022-07-01T21:57:48+0000 [INFO] [api_server:4] Service loaded from Bento directory: bentoml.Service(tag="iris_classifier:6otbsmxzq6lwbgxi", path="/home/bentoml/bento/")


Most of the deployment tools built on top of BentoML use Docker under the hood. It is
recommended to test out serving from a containerized Bento docker image first, before
moving to a production deployment. This helps verify the correctness of all the docker
and dependency configs specified in the :code:`bentofile.yaml`.


Deploying Bentos
----------------

BentoML standardizes the saved model format, service API definition and the Bento build
process, which opens up many different deployment options for ML teams.

The Bento we built and the docker image created in the previous steps are designed to
be DevOps friendly and ready for deployment in a production environment. If your team
has existing infrastructure for running docker, it's likely that the Bento generated
docker images can be directly deployed to your infrastructure without any modification.

.. note::

   To streamline the deployment process, BentoServer follows most common best practices
   found in a backend service: it provides
   :doc:`health check and prometheus metrics <guides/monitoring>`
   endpoints for monitoring out-of-the-box; It provides configurable
   :doc:`distributed tracing <guides/tracing>` and :doc:`logging <guides/logging>` for
   performance analysis and debugging; and it can be easily
   :doc:`integrated with other tools <integrations/index>` that are commonly used by
   Data Engineers and DevOps engineers.


For teams looking for an end-to-end solution, with more powerful deployment features
specific for ML, the BentoML team has also created Yatai and bentoctl:

.. grid::  1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 0

    .. grid-item-card:: `🦄️ Yatai <https://github.com/bentoml/Yatai>`_
        :link: https://github.com/bentoml/Yatai
        :link-type: url

        Model Deployment at scale on Kubernetes.

    .. grid-item-card:: `🚀 bentoctl <https://github.com/bentoml/bentoctl>`_
        :link: https://github.com/bentoml/bentoctl
        :link-type: url

        Fast model deployment on any cloud platform.

Learn more about different deployment options with BentoML from the
:doc:`concepts/deploy` page.


----

.. button-ref:: concepts/index
   :ref-type: doc
   :color: secondary
   :expand:

   Continue Reading
