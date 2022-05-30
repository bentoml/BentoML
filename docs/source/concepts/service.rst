================
Service and APIs
================

The service definition is the manifestation of the 
`Service Oriented Architecture <https://en.wikipedia.org/wiki/Service-oriented_architecture>`_ 
and the core building block in BentoML where users define the model serving logic. This
guide will dissect and explain the key components in the service definition.


Creating a Service
------------------

A BentoML service is composed of Runners and APIs. Consider the following service
definition from the :doc:`tutorial <tutorial>`:

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


Services are initialized through `bentoml.Service()` call, with the service name and a
list of :doc:`Runners </concepts/runner>` required in the service:

.. code:: python

    # Create the iris_classifier_service with the ScikitLearn runner
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

.. note::
    The service name will become the name of the Bento.

The :code:`svc` object created provides a decorator method :code:`svc.api`for defining
APIs in this service:

.. code:: python

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result


Runners
-------

Runners represent a unit of serving logic that can be scaled horizontally to maximize
throughput and resource utilization.

BentoML provides a convenient way of creating Runner instance from a saved model:

.. code:: python

    runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

.. tip::
    Users can also create custom Runners via the :doc:`Runner and Runnable interface <concepts/runner>`.


Runner created from a model will automatically choose the most optimal Runner
configurations specific for the target ML framework.

For example, if an ML framework releases the Python GIL and supports concurrent access
natively, BentoML will create a single global instance of the runner worker and route
all API requests to the global instance; otherwise, BentoML will create multiple
instances of runners based on the available system resources. We also let advanced users
to customize the runtime configurations to fine tune the runner performance. To learn
more, please see the :doc:`concepts/runner` guide.


Service APIs
------------

Inference APIs define how the service functionality can be accessed remotely. An API
consist of its input/output spec and a callback function:

.. code:: python

    # Create new API and add it to "svc"
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())  # define IO spec
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define business logic
        # Define pre-processing logic
        result = runner.run(input_array)  #  model inference call
        # Define post-processing logic
        return result

By decorating a function with :code:`@svc.api`, we declare that the function will be
invoked when this API is accessed. The API function is a great place to define your
serving logic, such as fetching additional features from your database, preprocess
features, and running inference via Runners.

When running :code:`bentoml serve` with the example above, this API definition
translates into an HTTP endpoint :code:`/predict`, that takes in a NumPy Ndarray
serialized as json format, and returns a NumPy Ndarray:

.. code:: bash

    > curl \
        -X POST \
        -H "content-type: application/json" \
        --data "[[5.9, 3, 5.1, 1.8]]" \
        http://127.0.0.1:3000/predict

    "[0]"

.. tip::
    BentoML also plan to support translating the same Service API definition into a gRPC
    server endpoint, in addition to the default HTTP server. See :issue:`703`.

By default, the function name becomes the endpoint URL. Users can also customize
this URL via the :code:`route` option, e.g.:

.. code:: python

    @svc.api(
        input=NumpyNdarray(),
        output=NumpyNdarray(),
        route="/v2/models/my_model/versions/v0/infer",
    )
    def predict(input_array: np.ndarray) -> np.ndarray:
        return runner.run(input_array)


A service can have one or many APIs. The :code:`input` and :code:`output` arguments of
the `@svc.api` decorator further defines the expect IO formats of the API. In the above
example, the API defines the IO types as :code:`numpy.ndarray` through a
:ref:`bentoml.io.NumpyNdarray<reference/api_io_descriptors:NumPy ndarray>` instance.


.. note::
    BentoML aims to parallelize API logic by starting multiple instances of the API
    server based on available system resources.


IO Descriptors
--------------

IO descriptors are used for defining an API's input and output specifications. It
describes the expected data type, helps validate that the input and output conform to
the expected format and schema and convert them from and to the native types. They are
specified through the :code:`input` and :code:`output` arguments in the :code:`@svc.api`
decorator method.

Recall the API we created in the :doc:`tutorial`. The classify API both accepts
arguments and returns results in the type of
:ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy ndarray>`:

.. code-block:: python

    import numpy as np
    from bentoml.io import NumpyNdarray

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_array: np.ndarray) -> np.ndarray:
        ...


Besides the :code:`NumpyNdarray` IO descriptor, BentoML supports a variety of IO
descriptors including :code:`PandasDataFrame`, :code:`JSON`, :code:`String`,
:code:`Image`, :code:`Text`, and :code:`File`. For detailed documentation on how to
declare and invoke these descriptors please see the
:doc:`IO Descriptors </reference/api_io_descriptors>` API reference page.


Schema and Validation
^^^^^^^^^^^^^^^^^^^^^

The IO descriptors help automatically generate an OpenAPI specifications of the service
based on the types of IO descriptors selected. We can further customize the IO
descriptors by providing the :code:`dtype` of the `numpy.ndarray` object. The provided
:code:`dtype` will be automatically translated in the generated OpenAPI specification.
The IO descriptors will validate the arguments and return values against the provided
:code:`dtype`. Requests that fail the validation will result in errors. We can choose to
optionally disable validation through the :code:`validate` argument.

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


Built-in Types
^^^^^^^^^^^^^^

Beside :code:`NumpyNdarray`, BentoML supports a variety of other built-in IO descriptor
types under the :doc:`bentoml.io <reference/api_io_descriptors>` module. Each type comes
with support of type validation and OpenAPI specification generation. For example:

+-----------------+---------------------+------------------+-------------------------+
| IO Descriptor   | Type                | Arguments        | Schema Type             |
+=================+=====================+==================+=========================+
| NumpyNdarray    | numpy.ndarray       | validate, schema | numpy.dtype             |
+-----------------+---------------------+------------------+-------------------------+
| PandasDataFrame | pandas.DataFrame    | validate, schema | pandas.DataFrame.dtypes |
+-----------------+---------------------+------------------+-------------------------+
| Json            | Python native types | validate, schema | Pydantic.BaseModel      |
+-----------------+---------------------+------------------+-------------------------+

Learn more about other built-in IO Descriptors :doc:`here <reference/api_io_descriptors>`.

Composite Types
^^^^^^^^^^^^^^^

The :code:`Multipart` IO descriptors can be used to group multiple IO Descriptor
instances, which allows the API function to accept multiple arguments or return multiple
values. Each IO descriptor can be customized with independent schema and validation
logic:

.. code-block:: python

    import typing as t
    import numpy as np
    from pydantic import BaseModel

    from bentoml.io import NumpyNdarray, Json

    class FooModel(BaseModel):
        field1: int
        field2: float
        field3: str

    my_np_input = NumpyNdarray.from_sample(np.ndarray(...))

    @svc.api(
        input=Multipart(
            arr=NumpyNdarray(schema=np.dtype(int, 4), validate=True),
            json=Json(pydantic_model=FooModel),
        )
        output=NumpyNdarray(schema=np.dtype(int), validate=True),
    )
    def predict(arr: np.ndarray, json: t.Dict[str, t.Any]) -> np.ndarray:
        ...



.. TODO:
    Document Open API (Swagger) generation and sample usage


Sync vs Async APIs
------------------

APIs can be defined as either synchronous function or asynchronous coroutines in Python.
The API we created in the :doc:`tutorial <tutorial>` was a synchronous API. BentoML will
intelligently create an optimally sized pool of workers to execute the synchronous
logic. Synchronous APIs are simple and capable of getting the job done for most model
serving scenarios.

.. code-block:: python

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_array: np.ndarray) -> np.ndarray:
        result = runner.run(input_array)
        return result

Synchronous APIs fall short when we want to maximize the performance and throughput of
the service. Asynchronous APIs are preferred if the processing logic is IO-bound or
invokes multiple runners simultaneously. The following async API example calls a remote
feature store asynchronously, invokes two runners simultaneously, and returns a combined
result.

.. code-block:: python

    import aiohttp
    import asyncio

    # Load two runners for two different versions of the ScikitLearn
    # Iris Classifier models we saved before
    runner1 = bentoml.sklearn.get("iris_clf:yftvuwkbbbi6zc").to_runner()
    runner2 = bentoml.sklearn.get("iris_clf:edq3adsfhzi6zg").to_runner()

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    async def predict(input_array: np.ndarray) -> np.ndarray:
        # Call a remote feature store to pre-process the request
        async with aiohttp.ClientSession() as session:
            async with session.get('https://features/get', params=input_array[0]) as resp:
                features = get_features(await resp.text())

        # Invoke both model runners simultaneously
        results = await asyncio.gather(
            runner1.predict.async_run(input_array, features),
            runner2.predict.async_run(input_array, features),
        )
        return combine_results(results)

The asynchronous API implementation is more efficient because when an asynchronous
method is invoked, the event loop is released to service other requests while this
request awaits the results of the method. In addition, BentoML will automatically
configure the ideal amount of parallelism based on the available number of CPU cores.
Further tuning of event loop configuration is not needed under common use cases.

