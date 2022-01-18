.. _api-io-descriptors-page:

API and IO Descriptors
======================

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
        result = await runner.run(input_array)
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

The asynchronous API implementation is more efficient because while the coroutine is awaiting for 
results from the feature store or the model runners, the event loop is freed up to serve another request. 
BentoML will intelligently create an optimally sized event loop based on the available number of CPU cores. Further tuning of event loop configuration is not needed under common use cases.

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

    Add further reading section
