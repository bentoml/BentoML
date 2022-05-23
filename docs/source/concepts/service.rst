================
Service and APIs
================

The service definition is the manifestation of the 
`Service Oriented Architecture <https://en.wikipedia.org/wiki/Service-oriented_architecture>`_ 
and the core building block in BentoML where users define the service runtime architecture and model serving logic. 
This guide will dissect and explain the key components in the service definition. By the end, you will gain a full 
understanding of the composition of the service definition and the responsibilities of each key component.

Composition
-----------

Consider the following service definition we created in the :ref:`Getting Started <getting-started-page>` guide. 
A BentoML service is composed of three components.
- Runners
- Services
- APIs

.. code-block:: python

    # bento.py
    import bentoml
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
should be parts of the service are passed in through the `runners` keyword argument. This is an important step because this is
how the BentoML library knows which runners to package into the bento. Build time and runtime behaviors of the service can be
customized through the `svc` instance.

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
`String`, `Image`, and `File`. For detailed documentation on how to declare and invoke these descriptors please see the :ref:`API Reference for IO descriptors <api-io-descriptors>`

The API is also a great place to define your pre- and post-process logic of model serving. In the example above, the logic defined
in the `predict` function will be packaged and deployed as a part of the serving logic.

BentoML aims to parallelize API logic by starting multiple instances of the API server based on available system resources. For
optimal performance, we recommend defining asynchronous APIs. To learn more, continue to :ref:`IO descriptors <api-io-descriptors-page>`.




API and IO Descriptors


APIs are functions defined in the service definition that are exposed as an HTTP or gRPC endpoint.
A function is a part of the APIs if it is decorated with the `@svc.api` decorator. APIs can be defined
either as a synchronous function or
`asynchronous coroutine <https://docs.python.org/3/library/asyncio-task.html>`_ in Python. APIs fulfill
requests by invoking the pre- and post-processing logic in the function and model runners created in the
service definition. Let's look into each of these parts in details.

Sync vs Async APIs
------------------

APIs can be defined as either synchronous function or asynchronous coroutines in Python. The API we
created in the :ref:`Getting Started <getting-started-page>`
guide was a synchronous API. BentoML will intelligently create an optimally sized pool of workers to
execute the synchronous logic. Synchronous APIs are simple and capable of getting the job done for many
common model serving scenarios.

.. code-block:: python

    # Create API function with pre- and post- processing logic
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = runner.run(input_array)
        # Define post-processing logic
        return result

Synchronous APIs fall short when we want to maximize the performance and throughput of the service.
Asynchronous APIs are preferred if the processing logic is IO-bound or invokes multiple runners
simultaneously. The following async API example calls a remote feature store asynchronously, invokes
two runners simultaneously, and returns the better result.

.. code-block:: python

    import aiohttp
    import asyncio

    # Load two runners for two different versions of the ScikitLearn
    # Iris Classifier models we saved before
    runner1 = bentoml.sklearn.load_runner("iris_classifier_model:yftvuwkbbbi6zcphca6rzl235")
    runner2 = bentoml.sklearn.load_runner("iris_classifier_model:edq3adsfhzi6zgr6vtpeqaare")

    # Create async API coroutine with pre-rocessing logic calling a feature store
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    async def predict(input_array: np.ndarray) -> np.ndarray:
    # Call a remote feature store to pre-process the request
        async with aiohttp.ClientSession() as session:
        params = [("key", v) for v in a]
        async with session.get('https://features/get', params=input_array[0]) as resp:
        features = get_features(await resp.text())
        # Invoke both model runners simultaneously and return the better result
        results = await asyncio.gather(
            runner1.async_run(input_array, features),
            runner2.async_run(input_array, features),
        )
        return compare_results(results)

The asynchronous API implementation is more efficient because when an asynchronous method is invoked, the event loop is
released to service other requests while this request awaits the results of the method. In addition, BentoML will automatically
configure the ideal amount of parallelism based on the available number of CPU cores. Further tuning of event loop configuration
is not needed under common use cases.

IO Descriptors
--------------

The input and output descriptors define the API specifications and validate the arguments and return
values of the API at runtime. They are specified through the `input` and `output` arguments in the
`@svc.api` decorator. Recall the API we created in the :ref:`Getting Started <getting-started-page>`
guide. The predict API both accepts arguments and returns results in the type of `bentoml.io.NumpyNdarray`.
`NumpyNdarray` describes the argument of return value of type `numpy.ndarray`, as specified in the Python
function signature.

.. code-block:: python

    import numpy as np

    from bentoml.io import NumpyNdarray

    # Create API function with pre- and post- processing logic
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = await runner.run(input_array)
        # Define post-processing logic
        return result

The IO descriptors help automatically generate an OpenAPI specifications of the service based on the
types of IO descriptors selected. We can further customize the IO descriptors by providing the `dtype`
of the `numpy.ndarray` object. The provided `dtype` will be automatically translated in the generated
OpenAPI specification. The IO descriptors will validate the arguments and return values against the
provided `dtype`. Requests that fail the validation will result in errors. We can choose to optionally
disable validation through the `validate` argument.

.. code-block:: python

    import numpy as np

    from bentoml.io import NumpyNdarray

    # Create API function with pre- and post- processing logic
    @svc.api(
        input=NumpyNdarray(schema=np.dtype(int, 4), validate=True),
        output=NumpyNdarray(schema=np.dtype(int), validate=True),
    )
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define pre-processing logic
        result = await runner.run(input_array)
        # Define post-processing logic
        return result

.. todo::

    insert Swagger screenshot

Built-in Types
--------------

Beside `NumpyNdarray`, BentoML supports a variety of other built-in IO descriptor types under the
`bentoml.io` package. Each type comes with support of type validation and OpenAPI specification
generation.

+-----------------+---------------------+------------------+-------------------------+
| IO Descriptor   | Type                | Arguments        | Schema Type             |
+=================+=====================+==================+=========================+
| NumpyNdarray    | numpy.ndarray       | validate, schema | numpy.dtype             |
+-----------------+---------------------+------------------+-------------------------+
| PandasDataFrame | pandas.DataFrame    | validate, schema | pandas.DataFrame.dtypes |
+-----------------+---------------------+------------------+-------------------------+
| Json            | Python native types | validate, schema | Pydantic.BaseModel      |
+-----------------+---------------------+------------------+-------------------------+

Composite Types
---------------

Multiple IO descriptors can be specified as tuples in the input and output arguments the API decorator.
Composite IO descriptors allow the API to accept multiple arguments and return multiple values. Each IO
descriptor can be customized with independent schema and validation logic.

.. code-block:: python

    import typing as t
    import numpy as np
    from pydantic import BaseModel

    from bentoml.io import NumpyNdarray, Json

    class FooModel(BaseModel):
        """Foo model documentation"""
        field1: int
        field2: float
        field3: str

    my_np_input = NumpyNdarray.from_sample(np.ndarray(...))

    # Create API function with pre- and post- processing logic
    @svc.api(
    input=Multipart(
        arr=NumpyNdarray(schema=np.dtype(int, 4), validate=True),
        json=Json(pydantic_model=FooModel),
    )
    output=NumpyNdarray(schema=np.dtype(int), validate=True),
    )
    def predict(arr: np.ndarray, json: t.Dict[str, t.Any]) -> np.ndarray:
        ...

.. todo::
    TODO: Open API



