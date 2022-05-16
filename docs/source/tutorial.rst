.. _tutorial-page:

Tutorial: Intro to BentoML
==========================

What Are We Building
--------------------
We will build a prediction service with BentoML, using an Iris classification model trained with Scikit-Learn.
By the end of this tutorial, we will have an HTTP endpoint for receiving inference requests and a docker container image
for deploying the model server.

.. note::
    You might be tempted to skip this tutorial because you are not using scikit-learn,
    but give it a chance. The concepts you will learn in the tutorial are fundamental to model serving
    with any ML framework using BentoML, and mastering it will give you a deep understanding of BentoML.


Setup for the tutorial
----------------------

There are two ways to complete this tutorial: you can either run the code in browser with Google Colab,
or you can set up a local development environment on your computer.

#. Run with Google Colab
    👉 `Open Tutorial Notebook on Colab <https://colab.research.google.com/github/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb>`_
    side by side with this guide. As you go through this guide, you can simply run the sample
    code from the Colab Notebook.

    You will be able to try out most of the content in the tutorial on Colab besides
    the docker container part towards the end. This is because Google Colab currently does not support docker.

#. Local Development Environment
    BentoML supports Linux, Windows and MacOS. Make sure you have Python 3.7 or above installed.
    We recommend using `virtual environment <https://docs.python.org/3/library/venv.html>`_ to create an isolated
    local environment for installing the Python dependencies required for the tutorial. However this is not required.

    You may download the source code of this tutorial from `bentoml/Gallery <https://github.com/bentoml/gallery/>`_:

    .. code-block:: bash

        git clone --depth=1 git@github.com:bentoml/gallery.git
        cd gallery/quickstart/


Installation
~~~~~~~~~~~~

You will need Python 3.7 or above to run this tutorial.

Install all Python packages required for this tutorial:

.. code-block:: bash

    pip install bentoml>=1.0.0a scikit-learn pandas


How does BentoML work?
----------------------

BentoML is a python-first, efficient and flexible framework for building
production-grade machine learning services.


It lets you save and version your trained models in a local model store, and defines a
standard interface for retrieving and running the saved models.



Save Model
----------

We begin by saving a trained model instance to BentoML's local
:ref:`model store <bento-management-page>`. The local model store is used for managing
your trained models as well as accessing them for serving.

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
    # INFO  [cli] Successfully saved Model(tag="iris_clf:7drxqvwsu6zq5uqj", path="~/bentoml/models/iris_clf/7drxqvwsu6zq5uqj/")


The model is now saved under name :code:`iris_clf` with an automatically generated
version.


:code:`bentoml.sklearn.save_model` is built specifically for the Scikit-Learn framework and uses its native saved model format under the hood for best compatibility and performance.
This goes the same for other ML framework, see :ref:`ML framework specific API <frameworks-page>` for other supported ML frameworks.

You can then load the the model to be run inline using the :code:`bentoml.<FRAMEWORK>.load(<TAG>)`

Or you can use our performance optimized runners using the :code:`bentoml.<FRAMEWORK>.load_runner(<TAG>)` API:

.. code-block:: python

    iris_clf_runner = bentoml.sklearn.load_runner("iris_clf:latest")
    iris_clf_runner.run(np.array([5.9, 3. , 5.1, 1.8]))

Models can also be managed via the :code:`bentoml models` CLI command. For more information use
:code:`bentoml models --help`.

.. code-block:: bash

    > bentoml models list iris_clf

    Tag                        Module           Path                                        Size      Creation Time
    iris_clf:svcryrt5xgafweb5  bentoml.sklearn  ~/bentoml/models/iris_clf/svcryrt5xgafweb5  5.81 KiB  2022-01-25 08:34:16


Define and Debug Services
-------------------------

Services are the core components of BentoML, where the serving logic is defined. With the model
saved in the model store, we can define the :ref:`service <service-definition-page>` by creating a
Python file :code:`service.py` with the following contents:

.. code-block:: python

    # service.py
    import numpy as np
    import bentoml
    from bentoml.io import NumpyNdarray

    # Load the runner for the latest ScikitLearn model we just saved
    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    # Create the iris_classifier service with the ScikitLearn runner
    # Multiple runners may be specified if needed in the runners array
    # When packaged as a bento, the runners here will included
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    # Create API function with pre- and post- processing logic with your new "svc" annotation
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = iris_clf_runner.predict.run(input_series)
        # Define post-processing logic
        return result

In this example, we defined the input and output type to be :code:`numpy.ndarray`. More options, such as
:code:`pandas.DataFrame` and :code:`PIL.image` are also supported. To see all supported options, see
:ref:`API and IO Descriptors <api-io-descriptors>`.

We now have everything we need to serve our first request. Launch the server in debug mode by
running the :code:`bentoml serve` command in the current working directory. Using the
:code:`--reload` option allows the server to reflect any changes made to the :code:`service.py` module
without restarting:

.. code-block:: bash

    > bentoml serve ./service.py:svc --reload

    02/24/2022 02:43:40 INFO     [cli] Starting development BentoServer from "./service.py:svc" running on http://127.0.0.1:3000 (Press CTRL+C to quit)                                                                                                                                                                   
    02/24/2022 02:43:41 INFO     [dev_api_server] Service imported from source: bentoml.Service(name="iris_classifier", import_str="service:svc", working_dir="/home/user/gallery/quickstart")                                                                                                                  
    02/24/2022 02:43:41 INFO     [dev_api_server] Will watch for changes in these directories: ['/home/user/gallery/quickstart']                                                                                                                                                                                
    02/24/2022 02:43:41 INFO     [dev_api_server] Started server process [25915]                                                                                                                                                                                                                                          
    02/24/2022 02:43:41 INFO     [dev_api_server] Waiting for application startup.                                                                                                                                                                                                                                        
    02/24/2022 02:43:41 INFO     [dev_api_server] Application startup complete.                                                                                                                          on.py:59

We can then send requests to the newly started service with any HTTP client:

.. tabs::

    .. code-tab:: python

        import requests
        requests.post(
            "http://127.0.0.1:3000/classify",
            headers={"content-type": "application/json"},
            data="[5,4,3,2]").text

    .. code-tab:: bash

        > curl \
          -X POST \
          -H "content-type: application/json" \
          --data "[5,4,3,2]" \
          http://127.0.0.1:3000/classify


BentoML optimizes your service in a number of ways for example we use two of the fastest Python web framework `Starlette <https://www.starlette.io/>`_ and `Uvicorn <https://www.uvicorn.org>`_, in order to serve your model efficiently at scale.

For more information on our performance optimizations please see :ref:`BentoServer <bento-server-page>`.

Build and Deploy Bentos
-----------------------

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

    02/24/2022 02:47:06 INFO     [cli] Building BentoML service "iris_classifier:dpijemevl6nlhlg6" from build context "/home/user/gallery/quickstart"                                                                                                                                                           
    02/24/2022 02:47:06 INFO     [cli] Packing model "iris_clf:tf773jety6jznlg6" from "/home/user//bentoml/models/iris_clf/tf773jety6jznlg6"                                                                                                                                                                            
    02/24/2022 02:47:06 INFO     [cli] Locking PyPI package versions..                                                                                                                                                                                                                                                    
    02/24/2022 02:47:08 INFO     [cli]                                                                                                                                                                                                                                                                                    
                                ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░                                                                                                                                                                                                                            
                                ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░                                                                                                                                                                                                                            
                                ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░                                                                                                                                                                                                                            
                                ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░                                                                                                                                                                                                                            
                                ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗                                                                                                                                                                                                                            
                                ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                        
    02/24/2022 02:47:08 INFO     [cli] Successfully built Bento(tag="iris_classifier:dpijemevl6nlhlg6") at "/home/user//bentoml/bentos/iris_classifier/dpijemevl6nlhlg6/"

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

    02/24/2022 03:01:19 INFO     [cli] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="/Users/ssheng/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")                                                                                                                                                                
    02/24/2022 03:01:19 INFO     [cli] Starting production BentoServer from "bento_identifier" running on http://0.0.0.0:3000 (Press CTRL+C to quit)                                                                                                                                                                                                                 
    02/24/2022 03:01:20 INFO     [iris_clf] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="/Users/ssheng/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")                                                                                                                                                           
    02/24/2022 03:01:20 INFO     [api_server] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="/Users/ssheng/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")                                                                                                                                                         
    02/24/2022 03:01:20 INFO     [iris_clf] Started server process [28761]                                                                                                                                                                                                                                                                                           
    02/24/2022 03:01:20 INFO     [iris_clf] Waiting for application startup.                                                                                                                                                                                                                                                                                         
    02/24/2022 03:01:20 INFO     [api_server] Started server process [28762]                                                                                                                                                                                                                                                                                         
    02/24/2022 03:01:20 INFO     [api_server] Waiting for application startup.                                                                                                                                                                                                                                                                                       
    02/24/2022 03:01:20 INFO     [api_server] Application startup complete.                                                                                                                                                                                                                                                                                          
    02/24/2022 03:01:20 INFO     [iris_clf] Application startup complete. 

