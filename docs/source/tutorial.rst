==========================
Tutorial: Intro to BentoML
==========================

BentoML is a python-first, efficient and flexible framework for machine learning model
serving. It lets data scientists to save and version trained models in a standardize
format and unifies how a saved model can be accessed for serving. This enables ML
engineers to easily use the saved models for building online prediction services or
batch inference jobs.

BentoML also helps with defining the APIs, environments and dependencies for running a
model, providing a build tool that encapsulates all model artifacts, source code and
dependencies into a self-contained format :code:`Bento`, which is designed to be DevOps
friendly and ready for production deployment - just like docker built for ML models.

What are we building
--------------------

In this tutorial, we will focus on online model serving with BentoML, using a
classification model trained with Scikit-Learn and the Iris dataset. By the end of this
tutorial, we will have an HTTP endpoint for receiving inference requests and a docker
container image for deployment.


.. note::
    You might be tempted to skip this tutorial because you are not using scikit-learn,
    but give it a chance. The concepts you will learn in the tutorial are fundamental to
    model serving with any ML framework using BentoML, and mastering it will give you a
    deep understanding of BentoML.


Setup for the tutorial
----------------------

There are two ways to complete this tutorial: you can either run the code in browser
with Google Colab, or you can set up a local development environment on your computer.

#. Run with Google Colab
    👉 `Open Tutorial Notebook on Colab <https://colab.research.google.com/github/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb>`_
    side by side with this guide. As you go through this guide, you can simply run the
    sample code from the Colab Notebook.

    You will be able to try out most of the content in the tutorial on Colab besides
    the docker container part towards the end. This is because Google Colab currently
    does not support docker.

#. Local Development Environment
    BentoML supports Linux, Windows and MacOS. Make sure you have Python 3.7 or above
    installed. We recommend using `virtual environment <https://docs.python.org/3/library/venv.html>`_
    to create an isolated local environment for installing the Python dependencies
    required for the tutorial. However this is not required.

    You may download the source code of this tutorial from `bentoml/Gallery <https://github.com/bentoml/gallery/>`_:

    .. code:: bash

        git clone --depth=1 git@github.com:bentoml/gallery.git
        cd gallery/quickstart/

..
   TODO: add #. Run tutorial notebook from Docker


Install Dependencies
~~~~~~~~~~~~~~~~~~~~

You will need Python 3.7 or above to run this tutorial.

Install all dependencies required for this tutorial:

.. code-block:: bash

    pip install --pre bentoml
    pip install scikit-learn pandas


Saving Models with BentoML
--------------------------

To begin with BentoML, you will need to save your trained models with BentoML API in
its local model store. The local model store is used for managing all your trained
models locally as well as accessing them for serving.

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
    bentoml.sklearn.save_model"iris_clf", clf)

    # INFO  [cli] Using default model signature `{"predict": {"batchable": False}}` for sklearn model
    # INFO  [cli] Successfully saved Model(tag="iris_clf:2uo5fkgxj27exuqj", path="~/bentoml/models/iris_clf/2uo5fkgxj27exuqj/")


The model is now saved under the name :code:`iris_clf` with an automatically generated
version. And the original model instance can be loaded back into memory via:

.. code-block::

   model = bentoml.sklearn.load_model("iris_clf:2uo5fkgxj27exuqj")
   # Alternatively, use `latest` to find the newest version
   model = bentoml.sklearn.load_model("iris_clf:latest")


The :code:`bentoml.sklearn.save_model` API is built specifically for the Scikit-Learn
framework and uses its native saved model format under the hood for best compatibility
and performance. This goes the same for other ML framework, e.g.
:code:`bentoml.pytorch.save_model`, see the
:ref:`ML Framework Specific Guide <frameworks/index>` for usage with other supported ML
frameworks.

Managing models
~~~~~~~~~~~~~~~

Saved models can be managed via the :code:`bentoml models` CLI command. Try
:code:`bentoml models --help`. to learn more.

.. tab:: List

   .. code-block:: bash

      > bentoml models list

      Tag                        Module           Size        Creation Time        Path
      iris_clf:2uo5fkgxj27exuqj  bentoml.sklearn  5.81 KiB    2022-05-19 08:36:52  ~/bentoml/models/iris_clf/2uo5fkgxj27exuqj
      iris_clf:nb5vrfgwfgtjruqj  bentoml.sklearn  5.80 KiB    2022-05-17 21:36:27  ~/bentoml/models/iris_clf/nb5vrfgwfgtjruqj

.. tab:: Get

   .. code-block:: bash

      > bentoml models get iris_clf:latest

      name: iris_clf
      version: 2uo5fkgxj27exuqj
      module: bentoml.sklearn
      labels: {}
      options: {}
      metadata: {}
      context:
        framework_name: sklearn
        framework_versions:
          scikit-learn: 1.1.0
        bentoml_version: 1.0.0
        python_version: 3.8.12
      signatures:
        predict:
          batchable: false
      api_version: v1
      creation_time: '2022-05-19T08:36:52.456990+00:00'

.. tab:: Import / Export

   .. code-block:: bash

      > bentoml models export iris_clf:latest .

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") exported to ./iris_clf-2uo5fkgxj27exuqj.bentomodel

   .. code-block:: bash

      > bentoml models import ./iris_clf-2uo5fkgxj27exuqj.bentomodel

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") imported

   .. note::

      Model can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
      example: :code:`bentoml models export iris_clf:latest s3://my_bucket/my_prefix/`

.. tab:: Push / Pull

   If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
   push local Models to Yatai, it provides APIs and Web UI for managing all Models
   created by your team and stores model files on cloud blob storage such as AWS S3,
   MinIO or GCS.

   .. code-block:: bash

      > bentoml models push iris_clf:latest

      Successfully pushed model "iris_clf:2uo5fkgxj27exuqj"                                                                                                                                                                                           │

   .. code-block:: bash

      > bentoml models pull iris_clf:latest

      Successfully pulled model "iris_clf:2uo5fkgxj27exuqj"

   .. image:: _static/img/yatai-model-detail.png
     :alt: Yatai Model Details UI

.. tab:: Delete

   .. code-block:: bash

      > bentoml models delete iris_clf:latest -y

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") deleted

.. note::

   If you have existing model saved to file on disk, you will need to load the model
   in a python session and then use BentoML's framework specific :code:`save_model`
   method to put it into the BentoML model store.

   However, we recommend always save the model with BentoML as soon as it finished
   training and validation. By putting the :code:`save_model` call to the end of your
   training pipeline, all your finalized models can be managed in one place and ready
   for inference.

   Learn more from the :doc:`concepts/model` doc.


Creating a Service
------------------

Services are the core components of BentoML, where the serving logic is defined. Create
a file :code:`service.py` with:

.. code-block:: python

    # service.py
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

.. code-block:: bash

    > bentoml serve ./service.py:svc --reload

    INFO [cli] Starting development BentoServer from "./service.py:svc" running on http://127.0.0.1:3000 (Press CTRL+C to quit)
    INFO [dev_api_server] Service imported from source: bentoml.Service(name="iris_classifier", import_str="service:svc", working_dir="/home/user/gallery/quickstart")
    INFO [dev_api_server] Will watch for changes in these directories: ['/home/user/gallery/quickstart']
    INFO [dev_api_server] Started server process [25915]
    INFO [dev_api_server] Waiting for application startup.
    INFO [dev_api_server] Application startup complete.                                                                                                                          on.py:59

Send prediction requests with an HTTP client:

.. tab:: Python

   .. code-block:: python

      import requests
      requests.post(
         "http://127.0.0.1:3000/classify",
         headers={"content-type": "application/json"},
         data="[[5.9, 3, 5.1, 1.8]]").text

.. tab:: Bash

   .. code-block:: bash

      curl \
        -X POST \
        -H "content-type: application/json" \
        --data "[[5.9, 3, 5.1, 1.8]]" \
        http://127.0.0.1:3000/classify

.. tab:: Browser

   Open http://127.0.0.1:3000 in your browser and send test request from the web UI.


Using Models in a Service
~~~~~~~~~~~~~~~~~~~~~~~~~

In the service definition, it should not be loading the model instance directly.
Instead, we use the :code:`bentoml.sklearn.get` API to get a reference to the entry
in local model store, and convert it into a Runner instance.



In BentoML, the recommended way of running ML model inference in serving is via Runner, which gives BentoML more flexibility in terms of how to schedule the inference computation, how to batch inference requests and take advantage of hardware resoureces available. Saved models can be loaded as Runner instance as shown below:


ref: model, runner, dynamic batching


Service API and IO Descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


learn more, ref: service, async endpoint, openapi(swagger) spec,


Running BentoServer
~~~~~~~~~~~~~~~~~~~

explain

service
--reload
--production

endpoints are automatically generated base on the service api and io descriptors


ref: , customize bentoserver , using async



custom endpoint, io descirptors docs,


.. note::

   BentoML optimizes your service in a number of ways for example we use two of the fastest Python web framework `Starlette <https://www.starlette.io/>`_ and `Uvicorn <https://www.uvicorn.org>`_, in order to serve your model efficiently at scale.



In this example, we defined the input and output type to be :code:`numpy.ndarray`. More options, such as
:code:`pandas.DataFrame` and :code:`PIL.image` are also supported. To see all supported options, see
:ref:`API and IO Descriptors <api-io-descriptors>`.

# Load the runner for the latest ScikitLearn model we just saved

# Create the iris_classifier service with the ScikitLearn runner
# Multiple runners may be specified if needed in the runners array
# When packaged as a bento, the runners here will included


We now have everything we need to serve our first request. Launch the server in debug mode by
running the :code:`bentoml serve` command in the current working directory. Using the
:code:`--reload` option allows the server to reflect any changes made to the :code:`service.py` module
without restarting:



Building a Bento
----------------

Once we are happy with the service definition, we can build the model and service into a
bento. Bentos are the distribution format for services, and contains all the information required to
run or deploy those services, such as models and dependencies. For more information about building
bentos, see :ref:`Building Bentos <building-bentos-page>`.

To build a Bento, first create a file named :code:`bentofile.yaml` in your project directory:

.. code-block:: yaml

    # bentofile.yaml
    service: "service.py:svc"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_VARIABLE_NAME>
    description: "file: ./README.md"
    labels:
        owner: bentoml-team
        stage: demo
    include:
     - "*.py"  # A pattern for matching which files to include in the bento
    python:
      packages:
       - scikit-learn  # Additional libraries to be included in the bento
       - pandas

Next, use the :code:`bentoml build` CLI command in the same directory to build a bento.

.. code-block:: bash

    > bentoml build

    INFO [cli] Building BentoML service "iris_classifier:dpijemevl6nlhlg6" from build context "/home/user/gallery/quickstart"
    INFO [cli] Packing model "iris_clf:tf773jety6jznlg6" from "/home/user/bentoml/models/iris_clf/tf773jety6jznlg6"
    INFO [cli] Locking PyPI package versions..
    INFO [cli]
         ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
         ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
         ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
         ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
         ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
         ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
                                                                                                                                                                                                                                                                                                                        
    INFO [cli] Successfully built Bento(tag="iris_classifier:dpijemevl6nlhlg6") at "~/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6/"

Bentos built will be saved in the local :ref:`bento store <bento-management-page>`, which you can
view using the :code:`bentoml list` CLI command.

.. code-block:: bash

    > bentoml list

    Tag                               Service      Path                                               Size       Creation Time
    iris_classifier:dpijemevl6nlhlg6  service:svc  ~/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6  19.46 KiB  2022-02-24 10:47:08

We can serve bentos from the bento store using the :code:`bentoml serve --production` CLI
command. Using the :code:`--production` option will serve the bento in production mode.

.. code-block:: bash

    > bentoml serve iris_classifier:latest --production

    INFO [cli] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="~/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")
    INFO [cli] Starting production BentoServer from "service.py:svc" running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    INFO [iris_clf] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="~/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")
    INFO [api_server] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="~/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")
    INFO [iris_clf] Started server process [28761]                                                                                                                                                                                                                                                                                           
    INFO [iris_clf] Waiting for application startup.                                                                                                                                                                                                                                                                                         
    INFO [api_server] Started server process [28762]                                                                                                                                                                                                                                                                                         
    INFO [api_server] Waiting for application startup.                                                                                                                                                                                                                                                                                       
    INFO [api_server] Application startup complete.                                                                                                                                                                                                                                                                                          
    INFO [iris_clf] Application startup complete. 


Managing Bentos
~~~~~~~~~~~~~~~

Bentos are the unit of deployment in BentoML, one of the most important artifact to keep
track of for your model deployment workflow. Similar to Models, Bentos built can be
managed via the :code:`bentoml` CLI command:

.. tab:: List

   .. code-block:: bash

      > bentoml list

      Tag                               Size        Creation Time        Path
      iris_classifier:nvjtj7wwfgsafuqj  16.99 KiB   2022-05-17 21:36:36  ~/bentoml/bentos/iris_classifier/nvjtj7wwfgsafuqj
      iris_classifier:jxcnbhfv6w6kvuqj  19.68 KiB   2022-04-06 22:02:52  ~/bentoml/bentos/iris_classifier/jxcnbhfv6w6kvuqj

.. tab:: Get

   .. code-block:: bash

      > bentoml get iris_classifier:latest

      service: service:svc
      name: iris_classifier
      version: nvjtj7wwfgsafuqj
      bentoml_version: 1.0.0a7.post49+g8353bb22
      creation_time: '2022-05-17T21:36:36.436878+00:00'
      labels:
        owner: bentoml-team
        project: gallery
      models:
      - tag: iris_clf:nb5vrfgwfgtjruqj
        module: bentoml.sklearn
        creation_time: '2022-05-17T21:36:27.656424+00:00'
      runners:
      - name: iris_clf
        runnable_type: SklearnRunnable
        models:
        - iris_clf:nb5vrfgwfgtjruqj
        resource_config:
          cpu: 4.0
          nvidia_gpu: 0.0
      apis:
      - name: classify
        input_type: NumpyNdarray
        output_type: NumpyNdarray

.. tab:: Import / Export

   .. code-block:: bash

      > bentoml export iris_classifier:latest .

      INFO [cli] Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") exported to ./iris_classifier-nvjtj7wwfgsafuqj.bento

   .. code-block:: bash

      > bentoml import ./iris_classifier-nvjtj7wwfgsafuqj.bento

      INFO [cli] Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") imported

   .. note::

      Bentos can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
      example: :code:`bentoml export iris_classifier:latest s3://my_bucket/my_prefix/`

.. tab:: Push / Pull

   If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
   push local Bentos to Yatai, it provides APIs and Web UI for managing all Bentos
   created by your team, stores Bento files on cloud blob storage such as AWS S3, MinIO
   or GCS, and automatically builds docker images when a new Bento was pushed.

   .. code-block:: bash

      > bentoml push iris_classifier:latest

      Successfully pushed Bento "iris_classifier:nvjtj7wwfgsafuqj"

   .. code-block:: bash

      > bentoml pull iris_classifier:nvjtj7wwfgsafuqj

      Successfully pulled Bento "iris_classifier:nvjtj7wwfgsafuqj"

   .. image:: _static/img/yatai-bento-repos.png
     :alt: Yatai Bento Repo UI

.. tab:: Delete

   .. code-block:: bash

      > bentoml delete iris_classifier:latest -y

      INFO [cli] Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") deleted


ref: learn more building bento

Serving a Bento
---------------

bentoml serve iris_classifier:latest --production



Generate a Docker image
-----------------------



Deploying the Bento
-------------------
