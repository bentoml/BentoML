.. _service-definition-page:

Service Definition
==================

The service definition is the manifestation of the 
`Service Oriented Architecture <https://en.wikipedia.org/wiki/Service-oriented_architecture>`_ 
and the core building block in BentoML where users define the service runtime architecture and model serving logic. 
This guide will dissect and explain the key components in the service definition. By the end, you will gain a full 
understanding of the composition of the service definition and the responsibilities of each key component.

Composition
-----------

Consider the following service definition we created in the :ref:`Getting Started <getting-started-page>` guide. 
A BentoML service is composed of three components.
- APIs
- Runners
- Services

.. code-block:: python

    # bento.py
    import bentoml
    import bentoml.sklearn
    import numpy as np

    from bentoml.io import NumpyNdarray

    # Load the runner for the latest ScikitLearn model we just saved
    runner = bentoml.sklearn.load_runner("iris_classifier_model:latest")

    # Create the iris_classifier_service with the ScikitLearn runner
    svc = bentoml.Service("iris_classifier_service", runners=[runner])

    # Create API function with pre- and post- processing logic
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = runner.run(input_array)
        # Define post-processing logic
        return result

APIs
----

Inference APIs define how the service functionality can be accessed remotely and the high level pre- and post-processing logic.

.. code-block:: python

    # Create API function with pre- and post- processing logic
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = runner.run(input_array)
        # Define post-processing logic
        return result

By decorating a function with `@svc.api`, we declare that the function is a part of the APIs that can be accessed remotely. 
A service can have one or many APIs. The `input` and `output` arguments of the `@svc.api` decorator further defines the expect 
IO formats of the API. In the above example, the API defines the IO types as `numpy.ndarray` through the `NumpyNdarray` 
:ref:`IO descriptors <api-io-descriptors-page>`. IO descriptors help validate that the input and output conform to the expected format 
and schema and convert them from and to the native types. BentoML supports a variety of IO descriptors including `PandasDataFrame`, 
`String`, `Image`, and `File`.

The API is also a great place to define your pre- and post-process logic of model serving. In the example above, the logic defined 
in the `predict` function will be packaged and deployed as a part of the serving logic.

BentoML aims to parallelize API logic by starting multiple instances of the API server based on available system resources. For 
optimal performance, we recommend defining asynchronous APIs. To learn more, continue to :ref:`IO descriptors <api-io-descriptors-page>`.

Runners
-------

Runners represent a unit of serving logic that can be scaled horizontally to maximize throughput.

.. code-block:: python

    # Load the runner for the latest ScikitLearn model we just saved
    runner = bentoml.sklearn.load_runner("iris_classifier_model:latest")

Runners can be created either by calling the `load_runner()` framework specific function or  decorating the implementation classes 
with the `@svc.runner` decorator. The framework specific functions will intelligently load runners with the most optimal 
configurations for the ML framework to achieve the most mechanical sympathy. For example, if an ML framework releases the Python 
GIL and supports concurrent access natively, BentoML will create a single global instance of the runner and route all API requests 
to the global instance; otherwise, BentoML will create multiple instances of runners based on the available system resources. 
Do not worry, we also let advanced users to customize the runtime configurations to fine tune the runner performance.

The argument to the `load_runner()` function is the name and the version of the model we saved before. Using the `latest` keyword 
will ensure load the latest version of the model. Load runner also declares to the builder that a specific model and version should 
be packaged into the bento when the service is built. Multiple runners can be defined in a service.

To learn more, please see the :ref:`Runner <runner-page>` advanced guide.

Services
--------

Services are composed of APIs and Runners and can be initialized through `bentoml.Service()`.

.. code-block:: python

    # Create the iris_classifier_service with the ScikitLearn runner
    svc = bentoml.Service("iris_classifier_service", runners=[runner])

The first argument of the service is the name which will become the name of the Bento after the service is built. Runners that 
should be parts of the service are passed in through the `runners` keyword argument. Build time and runtime behaviors of the 
service can be customized through the `svc` instance.

Further Reading
---------------
- :ref:`Runner <runner-page>`
- :ref:`Bento Server <bento-server-page>`
- :ref:`API and IO descriptors <api-io-descriptors-page>`
- :ref:`Serving Multiple Modles <multiple-models-page>`
- :ref:`Building Bentos <building-bentos-page>`
