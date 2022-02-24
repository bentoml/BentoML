.. _getting-started-page:

Getting Started
===============

In this guide we will show you how to create a local web service for your machine learning model(s). 
Then we will package that web service into a self contained package (Bento) which is ready for production deployment. 
The source code of this guide is avaialble in the `gallery <https://github.com/bentoml/gallery/tree/main/quickstart>`_ project.

There are three parts to the BentoML workflow.

#. :ref:`Save your Model <save-models-section>`
    Once model training is complete, use one of our tool specific frameworks to save your model in BentoML's standard format.
#. :ref:`Define your Service <define-and-debug-service-section>`
    Now that we've stored your model in our standard format, we will define the webservice which will host the model. In this definition, you can easily add Pre/Post processing code along with your model inference.
#. :ref:`Build and Deploy your Bento <build-and-deploy-bentos>`
    Finally, let BentoML build your deployable container (your bento) and assist you in deploying to your cloud service of choice


.. _save-models-section:

Installation
------------

BentoML is distributed as a Python package and can be installed from PyPI:

.. code-block:: bash

    pip install bentoml --pre

** The :code:`--pre` flag is required as BentoML 1.0 is still a preview release

Save Models
-----------

We begin by saving a trained model instance to BentoML's local
:ref:`model store <bento-management-page>`. The local model store is used to version your models as well as control which models are packaged with your bento.

If the models you wish to use are already saved to disk or available in a cloud repository, they can also be added to BentoML with the
:ref:`import APIs <bento-management-page>`.

.. code-block:: python

    import bentoml

    from sklearn import svm
    from sklearn import datasets

    # Load predefined training set to build an example model
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    # Call to bentoml.<FRAMEWORK>.save(<MODEL_NAME>, model)
    # In order to save to BentoML's standard format in a local model store
    bentoml.sklearn.save("iris_clf", clf)

    # [08:34:16 AM] INFO     Successfully saved Model(tag="iris_clf:svcryrt5xgafweb5",
    #                        path="/home/user/bentoml/models/iris_clf/svcryrt5xgafweb5/")
    # Tag(name='iris_clf', version='svcryrt5xgafweb5')


:code:`bentoml.sklearn.save()`, will save the Iris Classifier to a local model store managed by BentoML.
See :ref:`ML framework specific API <frameworks-page>` for all supported modeling libraries.

You can then load the the model to be run inline using the :code:`bentoml.<FRAMEWORK>.load(<TAG>)`

Or you can use our performance optimized runners using the :code:`bentoml.<FRAMEWORK>.load_runner(<TAG>)` API:

.. code-block:: python

    iris_clf_runner = bentoml.sklearn.load_runner("iris_clf:latest")
    iris_clf_runner.run(np.array([5.9, 3. , 5.1, 1.8]))

Models can also be managed via the :code:`bentoml models` CLI command. For more information use
:code:`bentoml models --help`.

.. code-block:: bash

    > bentoml models list iris_clf

    Tag                        Module           Path                                                 Size      Creation Time
    iris_clf:svcryrt5xgafweb5  bentoml.sklearn  /home/user/bentoml/models/iris_clf/svcryrt5xgafweb5  5.81 KiB  2022-01-25 08:34:16

.. _define-and-debug-service-section:

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
    iris_clf_runner = bentoml.sklearn.load_runner("iris_clf:latest")

    # Create the iris_classifier service with the ScikitLearn runner
    # Multiple runners may be specified if needed in the runners array
    # When packaged as a bento, the runners here will included
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    # Create API function with pre- and post- processing logic with your new "svc" annotation
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = iris_clf_runner.run(input_series)
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

    02/24/2022 02:43:40 INFO     [cli] Starting development BentoServer from "./service.py:svc" running on http://127.0.0.1:5000 (Press CTRL+C to quit)                                                                                                                                                                   
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
            "http://127.0.0.1:5000/classify",
            headers={"content-type": "application/json"},
            data="[5,4,3,2]").text

    .. code-tab:: bash

        > curl \
          -X POST \
          -H "content-type: application/json" \
          --data "[5,4,3,2]" \
          http://127.0.0.1:5000/classify

.. _build-and-deploy-bentos:

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
    service: "service.py:svc"  # A convention for locating your service: <YOUR_SERVICE_PY>:<YOUR_SERVICE_ANNOTATION>
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

    Tag                               Service      Path                                                        Size       Creation Time
    iris_classifier:dpijemevl6nlhlg6  service:svc  /home/user/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6  19.46 KiB  2022-02-24 10:47:08

We can serve bentos from the bento store using the :code:`bentoml serve --production` CLI
command. Using the :code:`--production` option will serve the bento in production mode.

.. code-block:: bash

    > bentoml serve iris_classifier:latest --production

    02/24/2022 03:01:19 INFO     [cli] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="/Users/ssheng/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")                                                                                                                                                                
    02/24/2022 03:01:19 INFO     [cli] Starting production BentoServer from "bento_identifier" running on http://0.0.0.0:5000 (Press CTRL+C to quit)                                                                                                                                                                                                                 
    02/24/2022 03:01:20 INFO     [iris_clf] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="/Users/ssheng/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")                                                                                                                                                           
    02/24/2022 03:01:20 INFO     [api_server] Service loaded from Bento store: bentoml.Service(tag="iris_classifier:dpijemevl6nlhlg6", path="/Users/ssheng/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6")                                                                                                                                                         
    02/24/2022 03:01:20 INFO     [iris_clf] Started server process [28761]                                                                                                                                                                                                                                                                                           
    02/24/2022 03:01:20 INFO     [iris_clf] Waiting for application startup.                                                                                                                                                                                                                                                                                         
    02/24/2022 03:01:20 INFO     [api_server] Started server process [28762]                                                                                                                                                                                                                                                                                         
    02/24/2022 03:01:20 INFO     [api_server] Waiting for application startup.                                                                                                                                                                                                                                                                                       
    02/24/2022 03:01:20 INFO     [api_server] Application startup complete.                                                                                                                                                                                                                                                                                          
    02/24/2022 03:01:20 INFO     [iris_clf] Application startup complete. 

Lastly, we can :ref:`containerize bentos as Docker images <containerize-bentos-page>` using the
:code:`bentoml container` CLI command and manage bentos at scale using the
:ref:`model and bento management <bento-management-page>` service.

Further Reading
---------------
- :ref:`Containerize Bentos as Docker Images <containerize-bentos-page>`
- :ref:`Model and Bento Management <bento-management-page>`
- :ref:`Service Definition <service-definition-page>`
- :ref:`Building Bentos <building-bentos-page>`

.. spelling::
