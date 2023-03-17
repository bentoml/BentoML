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
definition from the :doc:`tutorial </tutorial>`:

.. code-block:: python

    import numpy as np
    import bentoml
    from bentoml.io import NumpyNdarray

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result


Services are initialized through ``bentoml.Service()`` call, with the service name and a
list of :doc:`Runners </concepts/runner>` required in the service:

.. code-block:: python

    # Create the iris_classifier_service with the ScikitLearn runner
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

.. note::
    The service name will become the name of the Bento.

The ``svc`` object created provides a decorator method ``svc.api`` for defining`
APIs in this service:

.. code-block:: python

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result


Runners
-------

Runners represent a unit of serving logic that can be scaled horizontally to maximize
throughput and resource utilization.

BentoML provides a convenient way of creating Runner instance from a saved model:

.. code-block:: python

    runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

.. tip::
    Users can also create custom Runners via the :doc:`Runner and Runnable interface </concepts/runner>`.


Runner created from a model will automatically choose the most optimal Runner
configurations specific for the target ML framework.

For example, if an ML framework releases the Python GIL and supports concurrent access
natively, BentoML will create a single global instance of the runner worker and route
all API requests to the global instance; otherwise, BentoML will create multiple
instances of runners based on the available system resources. We also let advanced users
to customize the runtime configurations to fine tune the runner performance. To learn
more, see the :doc:`introduction to Runners </concepts/runner>`.

Debugging Runners
^^^^^^^^^^^^^^^^^

Runners must be initialized in order to function. Normally, this is handled by BentoML internally
when ``bentoml serve`` is called.

If you want to import and run a service without using BentoML, this must be done manually. For
example, to debug a service called ``svc`` in ``service.py``:

.. code-block:: python

    from service import svc

    for runner in svc.runners:
        runner.init_local()

    result = svc.apis["my_endpoint"].func(inp)


Service APIs
------------

Inference APIs define how the service functionality can be called remotely. A service can 
have one or more APIs. An API consists of its input/output specs and a callback function:

.. code-block:: python

    # Create new API and add it to "svc"
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())  # define IO spec
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define business logic
        # Define pre-processing logic
        result = runner.run(input_array)  #  model inference call
        # Define post-processing logic
        return result

By decorating a function with ``@svc.api``, we declare that the function shall be
invoked when this API is called. The API function is a great place for defining your
serving logic, such as feature fetching, pre and post processing, and model inferences 
via Runners.

When running ``bentoml serve`` with the example above, this API function is
transformed into an HTTP endpoint, ``/predict``, that takes in a ``np.ndarray`` as 
input, and returns a ``np.ndarray`` as output. The endpoint can be called with the following
``curl`` command:

.. code-block:: bash

    » curl -X POST \
        -H "content-type: application/json" \
        --data "[[5.9, 3, 5.1, 1.8]]" \
        http://127.0.0.1:3000/predict

    "[0]"

.. tip::
    BentoML also plan to support translating the same Service API definition into a gRPC
    server endpoint, in addition to the default HTTP server. See :issue:`703`.

Route
^^^^^

By default, the function name becomes the endpoint URL. Users can also customize
this URL via the ``route`` option, e.g.:

.. code-block:: python

    @svc.api(
        input=NumpyNdarray(), output=NumpyNdarray(),
        route="/v2/models/my_model/versions/v0/infer",
    )
    def predict(input_array: np.ndarray) -> np.ndarray:
        return runner.run(input_array)


.. note::
    BentoML aims to parallelize API logic by starting multiple instances of the API
    server based on available system resources.

Inference Context
^^^^^^^^^^^^^^^^^

The context of an inference call can be accessed through the additional ``bentoml.Context``
argument added to the service API function. Both the request and response contexts can be 
accessed through the inference context for getting and setting the headers, cookies, and
status codes.

.. code-block:: python

    @svc.api(
        input=NumpyNdarray(),
        output=NumpyNdarray(),
    )
    def predict(input_array: np.ndarray, ctx: bentoml.Context) -> np.ndarray:
        # get request headers
        request_headers = ctx.request.headers

        result = runner.run(input_array)

        # set response headers, cookies, and status code 
        ctx.response.status_code = 202
        ctx.response.cookies = [
            bentoml.Cookie(
                key="key",
                value="value",
                max_age=None,
                expires=None,
                path="/predict",
                domain=None,
                secure=True,
                httponly=True,
                samesite="None"
            )
        ]
        ctx.response.headers.append("X-Custom-Header", "value")

        return result


IO Descriptors
--------------

IO descriptors are used for defining an API's input and output specifications. It
describes the expected data type, helps validate that the input and output conform to
the expected format and schema and convert them from and to the native types. They are
specified through the ``input`` and ``output`` arguments in the ``@svc.api``
decorator method.

Recall the API we created in the :doc:`tutorial </tutorial>`. The ``classify`` API both accepts
arguments and returns results in the type of
:ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`:

.. code-block:: python

    import numpy as np
    from bentoml.io import NumpyNdarray

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_array: np.ndarray) -> np.ndarray:
        ...


Besides the ``NumpyNdarray`` IO descriptor, BentoML supports a variety of IO
descriptors including ``PandasDataFrame``, ``JSON``, ``String``,
``Image``, ``Text``, and ``File``. For detailed documentation on how to
declare and invoke these descriptors please see the
:doc:`IO Descriptors </reference/api_io_descriptors>` API reference page.


Schema and Validation
^^^^^^^^^^^^^^^^^^^^^

IO descriptors allow users to define the expected data types, shape, and schema, based 
on the type of the input and output descriptor specified. IO descriptors can also be defined 
through  examples with the ``from_sample`` API to simplify the development of service 
definitions.

Numpy
~~~~~

The data type and shape of the ``NumpyNdarray`` can be specified with the ``dtype`` 
and ``shape`` arguments. By setting the ``enforce_shape`` and ``enforce_dtype`` 
arguments to `True`, the IO descriptor will strictly validate the input and output data 
based the specified data type and shape. To learn more, see IO descrptor reference for 
:ref:`reference/api_io_descriptors:NumPy ``ndarray```.

.. code-block:: python

    import numpy as np

    from bentoml.io import NumpyNdarray

    svc = bentoml.Service("iris_classifier")

    # Define IO descriptors through samples
    output_descriptor = NumpyNdarray.from_sample(np.array([[1.0, 2.0, 3.0, 4.0]]))

    @svc.api(
        input=NumpyNdarray(
            shape=(-1, 4),
            dtype=np.float32,
            enforce_dtype=True,
            enforce_shape=True
        ),
        output=output_descriptor,
    )
    def classify(input_array: np.ndarray) -> np.ndarray:
        ...

Pandas DataFrame
~~~~~~~~~~~~~~~~

The data type and shape of the ``PandasDataFrame`` can be specified with the ``dtype`` 
and ``shape`` arguments. By setting the ``enforce_shape`` and ``enforce_dtype`` 
arguments to `True`, the IO descriptor will strictly validate the input and output data 
based the specified data type and shape. To learn more, see IO descrptor reference for 
:ref:`reference/api_io_descriptors:Tabular Data with Pandas`.

.. code-block:: python

    import pandas as pd

    from bentoml.io import PandasDataFrame

    svc = bentoml.Service("iris_classifier")

    # Define IO descriptors through samples
    output_descriptor = PandasDataFrame.from_sample(pd.DataFrame([[5,4,3,2]]))

    @svc.api(
        input=PandasDataFrame(
            orient="records",
            dtype=np.float32,
            enforce_dtype=True,
            shape=(-1, 4),
            enforce_shape=True
        ),
        output=output_descriptor,
    )
    def classify(input_series: pd.DataFrame) -> pd.DataFrame:
        ...

JSON
~~~~

The data type of a JSON IO descriptor can be specified through a Pydantic model. By setting 
a pydantic model, the IO descriptor will validate the input based on the specified pydantic
model and return. To learn more, see IO descrptor reference for
:ref:`reference/api_io_descriptors:Structured Data with JSON`. We also provide
:examples:`an example project <pydantic_validation>` using Pydantic for request validation.

.. code-block:: python

    from typing import Dict, Any
    from pydantic import BaseModel

    svc = bentoml.Service("iris_classifier")

    class IrisFeatures(BaseModel):
        sepal_length: float
        sepal_width: float
        petal_length: float
        petal_width: float

    @svc.api(
        input=JSON(pydantic_model=IrisFeatures),
        output=JSON(),
    )
    def classify(input_series: IrisFeatures) -> Dict[str, Any]:
        input_df = pd.DataFrame([input_data.dict()])
        results = iris_clf_runner.predict.run(input_df).to_list()
        return {"predictions": results}


Built-in Types
^^^^^^^^^^^^^^

Beside ``NumpyNdarray``, BentoML supports a variety of other built-in IO descriptor
types under the :doc:`bentoml.io </reference/api_io_descriptors>` module. Each type comes
with support of type validation and OpenAPI specification generation. For example:

+-----------------+---------------------+---------------------+-------------------------+
| IO Descriptor   | Type                | Arguments           | Schema Type             |
+=================+=====================+=====================+=========================+
| NumpyNdarray    | numpy.ndarray       | validate, schema    | numpy.dtype             |
+-----------------+---------------------+---------------------+-------------------------+
| PandasDataFrame | pandas.DataFrame    | validate, schema    | pandas.DataFrame.dtypes |
+-----------------+---------------------+---------------------+-------------------------+
| Json            | Python native types | validate, schema    | Pydantic.BaseModel      |
+-----------------+---------------------+---------------------+-------------------------+
| Image           | PIL.Image.Image     | pilmodel, mime_type |                         |
+-----------------+---------------------+---------------------+-------------------------+
| Text            | str                 |                     |                         |
+-----------------+---------------------+---------------------+-------------------------+
| File            | BytesIOFile         | kind, mime_type     |                         |
+-----------------+---------------------+---------------------+-------------------------+

Learn more about other built-in IO Descriptors :doc:`here </reference/api_io_descriptors>`.

Composite Types
^^^^^^^^^^^^^^^

The ``Multipart`` IO descriptors can be used to group multiple IO Descriptor
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


Sync vs Async APIs
------------------

APIs can be defined as either synchronous function or asynchronous coroutines in Python.
The API we created in the :doc:`tutorial </tutorial>` was a synchronous API. BentoML will
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

.. tip::
    Blocking logic such as communicating with an API or database without the `await`
    keyword will block the event loop and prevent it from completing other IO tasks.
    If you must use a library that does not support asynchronous IO with `await`, you
    should use the synchronous API instead. If you are not sure, also use the synchronous
    API to prevent unexpected errors.


.. TODO:

    Running Server:
        bentoml serve arguments
        --reload
        --production

        other options and configs:
        --api-workers
        --backlog
        --timeout
        --host
        --port

        Config options:
        --config

    Endpoints:
        List of Endpoints
            POST: /{api_name}
        Open API (Swagger) generation and sample usage

    Exception handling
        custom error code
        custom error msg



