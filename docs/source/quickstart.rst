.. _getting-started-page:

Getting Started
***************

There are three parts to the BentoML workflow.

#. :ref:`Save Models <save-models-section>`
#. :ref:`Define and Debug Services <define-and-debug-service-section>`
#. :ref:`Build and Deploy Bentos <build-and-deploy-bentos>`

.. _save-models-section:

Save Models
-----------

We start with saving a trained model instance to BentoML's local 
:ref:`model store <bento-management-page>`. 
If models are already saved to file, they can also be brought to BentoML with the 
:ref:`import APIs <bento-management-page>`.

.. code-block:: python

    from sklearn import svm
    from sklearn import datasets

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    import bentoml.sklearn
    bentoml.sklearn.save("iris_classifier_model", clf)
    # [INFO] Scikit-learn model 'iris_classifier_model:yftvuwkbbbi6zcphca6rzl235' is successfully saved to BentoML local model store under "~/bentoml/models/iris_classifier_model/yftvuwkbbbi6zcphca6rzl235"

The :ref:`ML framework specific API <frameworks-page>`, `bentoml.sklearn.save()`, will save the Iris Classifier to a 
local model store managed by BentoML. And the `load_runner()` API can be used to load this model into a Runner:

.. code-block:: python

    iris_clf_runner = bentoml.sklearn.load_runner("iris_classifier_model:latest")
    iris_clf_runner.run(np.ndarray([5.9, 3. , 5.1, 1.8]))

Models can also be managed via the `bentoml models` CLI command, see `bentoml models --help` for more.

.. code-block:: bash

    > bentoml models list iris_classifier_model

    MODEL                 VERSION                   FRAMEWORK    CREATED            
    iris_classifier_model yftvuwkbbbi6zcphca6rzl235 ScikitLearn  2021/9/19 10:13:35

.. _define-and-debug-service-section:

Define and Debug Services
-------------------------

Services are the core components of BentoML where the serving logic is defined. With the model saved in the model store, 
we can define the :ref:`service <service-definition-page>` by creating a Python file `bento.py` in the working directory 
with the following contents. In the example below, we defined `numpy.ndarray` as the input and output type. More options 
like `pandas.dataframe` and `PIL.image` are also supported IO types, see @API and IO Descriptors.

.. code-block:: python

    # bento.py
    import bentoml
    import bentoml.sklearn
    import numpy as np

    from bentoml.io import NumpyNdarray

    # Load the runner for the latest ScikitLearn model we just saved
    iris_clf_runner = bentoml.sklearn.load_runner("iris_classifier_model:latest")

    # Create the iris_classifier_service with the ScikitLearn runner
    svc = bentoml.Service("iris_classifier_service", runners=[iris_clf_runner])


    # Create API function with pre- and post- processing logic
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_ndarray: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
    result = iris_clf_runner.run(input_ndarray)
    # Define post-processing logic
    return result

We now have everything needed to serve our first request, launch the server in debug mode by running the `bentoml serve` 
command in the current working directory. Using the `--reload` option allows the server to reflect any change in the 
`bento.py` module without restarting the server.

.. code-block:: bash

    > bentoml serve ./bento.py:svc --reload

    (Press CTRL+C to quit)
    [INFO] Starting BentoML API server in development mode with auto-reload enabled
    [INFO] Serving BentoML Service "iris_classifier_service" defined in "bento.py"
    [INFO] API Server running on http://0.0.0.0:5000

We can send requests to the newly started service with any clients.

.. tabs::

    .. code-tab:: python

        import requests
        requests.post(
            "http://0.0.0.0:5000/predict",
            headers={"content-type": "application/json"},
            data="[[5,4,3,2]]").text

    .. code-tab:: bash

        > curl \
        -X POST \
        -H "content-type: application/json" \
        --data "[[5, 4, 3, 2]]" \
        http://0.0.0.0:5000/predict

.. _build-and-deploy-bentos:

Build and Deploy Bentos
-----------------------

Once we are happy with the service definition, we can :ref:`build <building-bentos-page>` the model and service into a bento. 
Bentos are the distribution format of the service that can be deployed and contains all the information required for running 
the service, from models to the dependencies. By default, the built will automatically infer the dependent PyPI packages by 
recursive walking through all the modules. Use the `bentoml build` CLI command in the current working directory to build a bento.

.. code-block:: bash

    > bentoml build ./bento.py:svc
    
    [INFO] Building BentoML Service "iris_classifier_service" with models "iris_classifier_model:yftvuwkbbbi6zcphca6rzl235"
    [INFO] Bento is successfully built and saved to ~/bentoml/bentos/iris_classifier_model/v5mgcacfgzi6zdz7vtpeqaare

Bentos built will be saved in the local :ref:`bento store <bento-management-page>`, which we can view via the `bentoml list` CLI command.

.. code-block:: bash

    > bentoml list
    BENTO                   VERSION                    LABELS      CREATED
    iris_classifier_service v5mgcacfgzi6zdz7vtpeqaare  iris,prod   2021/09/19 10:15:50

We can serve bentos from the bento store using the `bentoml serve --production` CLI command. Using the `--production` option allows 
serving the bento in production mode.

.. code-block:: bash

    > bentoml serve iris_classifier_service:latest --production

    (Press CTRL+C to quit)
    [INFO] Starting BentoML API server in production mode
    [INFO] Serving BentoML Service "iris_classifier_service"
    [INFO] API Server running on http://0.0.0.0:5000

Lastly, we can :ref:`containerize bentos as Docker images <containerize-bentos-page>` using the `bentoml container` CLI command and manage 
Bentos at scale using the :ref:`model and bento management <bento-management-page>` service.

Further Reading
---------------
- :ref:`Containerize Bentos as Docker Images <containerize-bentos-page>`
- :ref:`Model and Bento Management <bento-management-page>`
- :ref:`Service Definition <service-definition-page>`
- :ref:`Building Bentos <building-bentos-page>`

.. spelling::

