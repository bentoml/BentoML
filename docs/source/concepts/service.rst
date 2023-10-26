========
Services
========

BentoML Services follow the principles of `Service Oriented Architecture <https://en.wikipedia.org/wiki/Service-oriented_architecture>`_,
serving as the core building blocks in BentoML. A Service allows you to define the serving logic of your model.

This page explains the key components in a BentoML Service.

Structure
---------

Key components of a BentoML Service include Runners and APIs, defined in a ``service.py`` file.
See the following Service definition example from :doc:`/quickstarts/deploy-an-iris-classification-model-with-bentoml`:

.. code-block:: python
   :caption: `service.py`

   import numpy as np
   import bentoml
   from bentoml.io import NumpyNdarray

   iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

   svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

   @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
   def classify(input_series: np.ndarray) -> np.ndarray:
       result = iris_clf_runner.predict.run(input_series)
       return result

You initialize a Service through the ``bentoml.Service()`` function, specifying its name and a list of :doc:`Runners </concepts/runner>`.
The Service name will become the name of the resulting Bento.

.. code-block:: python

    # Create the iris_classifier_service with the ScikitLearn Runner
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

The ``svc`` object created provides a decorator method ``svc.api`` for defining APIs in this Service.
This is where the logic to process data input and output is defined.

.. code-block:: python

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result

Details of each component are listed in the sections below.

Runners
-------

Runners are computation units that can be scaled horizontally to maximize throughput and optimize resource utilization.
BentoML provides a convenient way of instantiating a Runner from a saved model via ``to_runner()``.

.. code-block:: python

    runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

.. tip::

    You can also create custom Runners via the :doc:`Runner and Runnable interface </concepts/runner>`.

For the Runner created from a model, BentoML automatically chooses the optimal Runner
configurations specific for the target ML framework.

For example, if an ML framework releases the Python GIL and supports concurrent access
natively, BentoML will create a single global instance of the Runner worker and route
all API requests to the global instance; otherwise, BentoML will create multiple
instances of Runners based on the available system resources. BentoML supports customizing
the runtime configurations to fine tune the Runner performance. To learn more, see :doc:`/concepts/runner`.

Debugging Runners
^^^^^^^^^^^^^^^^^

Runners must be initialized in order to function. Normally, this is handled by BentoML internally
when ``bentoml serve`` is called, but you can also manually initialize and run a Service. For
example, to debug a Service called ``svc`` in ``service.py``, use the following code:

.. code-block:: python

    from service import svc

    for runner in svc.runners:
        runner.init_local()

    result = svc.apis["my_endpoint"].func(inp)

Service APIs
------------

Inference APIs define how the Service functionality can be called remotely. A Service can
have one or multiple APIs. An API consists of its IO specifications and a callback function:

.. code-block:: python

    # Create a new API and add it to "svc"
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())  # Define IO spec
    def predict(input_array: np.ndarray) -> np.ndarray:
        # Define business logic
        # Define pre-processing logic
        result = runner.run(input_array)  #  Model inference call
        # Define post-processing logic
        return result

By using the ``@svc.api`` decorator on a function, you specify that this function will be triggered when the API is called.
This API function serves as an ideal place to define your serving logic, encompassing tasks like feature fetching, pre-processing and post-processing,
as well as model inference via Runners.

When executed with ``bentoml serve`` using the example above, this API function is
transformed into an HTTP endpoint, or ``/predict`` in this case, which takes in a ``np.ndarray`` as
input, and returns a ``np.ndarray`` as output. The endpoint can be called with the following
``curl`` command:

.. code-block:: bash

    $ curl -X POST \
        -H "content-type: application/json" \
        --data "[[5.9, 3, 5.1, 1.8]]" \
        http://127.0.0.1:3000/predict

    "[0]"

Custom route paths
^^^^^^^^^^^^^^^^^^

By default, the function name becomes the endpoint URL. You can customize this URL via the ``route`` option:

.. code-block:: python

    @svc.api(
        input=NumpyNdarray(), output=NumpyNdarray(),
        route="/v2/models/my_model/versions/v0/infer",
    )
    def predict(input_array: np.ndarray) -> np.ndarray:
        return runner.run(input_array)

Inference context
^^^^^^^^^^^^^^^^^

You can retrieve the context of an inference call by adding ``bentoml.Context`` to the Service API function.
This parameter allows you to access information about the incoming request (like client headers)
and also modify the outgoing response (like setting response headers, cookies, or HTTP status codes).
Additionaly, you can read and write to the global state dictionary via the
``ctx.state`` attribute, which is a per-worker dictionary that can be read and written across
API endpoints.

.. code-block:: python

    @svc.api(
        input=NumpyNdarray(),
        output=NumpyNdarray(),
    )
    def predict(input_array: np.ndarray, ctx: bentoml.Context) -> np.ndarray:
        # Get request headers
        request_headers = ctx.request.headers

        result = runner.run(input_array)

        # Set response headers, cookies, and status code
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

        # Add a custom header to the response
        ctx.response.headers.append("X-Custom-Header", "value")

        return result

Lifecycle hooks
^^^^^^^^^^^^^^^

BentoML provides a set of lifecycle hooks, allowing Services to run specific code sequences at key moments, such as Service startup and shutdown.
Within these hooks, it's possible to access the inference context mentioned above. See the following example.

.. code-block:: python

    @svc.on_startup
    async def connect_db_on_startup(context: bentoml.Context):
        context.state["db"] = await get_db_connection()
        # ctx.request  # this will raise an error because no request has been served yet.

    @svc.on_shutdown
    async def close_db_on_shutdown(context: bentoml.Context):
        await context.state["db"].close()

The ``on_startup`` and ``on_shutdown`` hooks are triggered for each individual API server process (worker). We recommend you avoid
directly accessing the file system within these hooks to prevent potential conflicts. Instead, they're commonly used for initializing in-process resources, like establishing database connections.

Additionally, BentoML provides an ``on_deployment`` hook, which is activated just once when the Service starts.
This hook can be used for tasks such as downloading model files that should be accessible to all API server processes (workers).

.. code-block:: python

    @svc.on_deployment
    def download_model_on_serve():
        download_model_files()

This particular hook is executed on ``bentoml serve``, and it precedes the initiation of any worker processes.
You cannot access the inference context within the ``on_deployment`` hook.

.. note::

    While the ``on_deployment`` hook can run each time the Service starts, we still recommend you place any
    one-time initialization tasks in the :ref:`Setup Script <concepts/bento:Setup script>` to avoid repeated execution.

You can register multiple functions for each hook, and they will be executed in the order they are registered.
All hooks support both synchronous and asynchronous functions.

.. _io-descriptors:

IO descriptors
--------------

IO descriptors are used for defining an API's input and output specifications. They
describe the expected data type, help validate that the input and output conform to
the expected format and schema, and convert them from and to the native types. To use IO descriptors,
you need to import them from the ``bentoml.io`` package and specify them
through ``input`` and ``output`` in the ``@svc.api`` decorator.

In the following example, the ``classify`` API both accepts arguments and returns results in the type of
:ref:`bentoml.io.NumpyNdarray <reference/api_io_descriptors:NumPy \`\`ndarray\`\`>`:

.. code-block:: python

    import numpy as np
    from bentoml.io import NumpyNdarray

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_array: np.ndarray) -> np.ndarray:
        ...

BentoML supports a variety of built-in IO descriptors within the :doc:`bentoml.io </reference/api_io_descriptors>` module, including ``NumpyNdarray``, ``PandasDataFrame``, ``JSON``, ``Image``, ``Text``, and ``File``.
All these IO descriptors in BentoML support data validation and can generate OpenAPI specifications.

+-----------------+---------------------+---------------------+-------------------------+
| IO Descriptor   | Type                | Arguments           | Schema Type             |
+=================+=====================+=====================+=========================+
| NumpyNdarray    | numpy.ndarray       | validate, schema    | numpy.dtype             |
+-----------------+---------------------+---------------------+-------------------------+
| PandasDataFrame | pandas.DataFrame    | validate, schema    | pandas.DataFrame.dtypes |
+-----------------+---------------------+---------------------+-------------------------+
| JSON            | Python native types | validate, schema    | Pydantic.BaseModel      |
+-----------------+---------------------+---------------------+-------------------------+
| Image           | PIL.Image.Image     | pilmodel, mime_type |                         |
+-----------------+---------------------+---------------------+-------------------------+
| Text            | str                 |                     |                         |
+-----------------+---------------------+---------------------+-------------------------+
| File            | BytesIOFile         | kind, mime_type     |                         |
+-----------------+---------------------+---------------------+-------------------------+

Schema and validation
^^^^^^^^^^^^^^^^^^^^^

When you use IO descriptors, it is important to consider schema and data validation. This can prevent unexpected errors or issues due to data format mismatches.
You can define IO descriptors through examples with the ``from_sample`` API to simplify the development of Service
definitions.

The following sections provide some common examples of using IO descriptors with data validation.

NumPy
~~~~~

The ``NumpyNdarray`` IO descriptor is used for handling NumPy arrays. You specify its data type and shape with the ``dtype``
and ``shape`` arguments. By setting the ``enforce_shape`` and ``enforce_dtype``
arguments to ``True``, the IO descriptor strictly validates the input and output data
based on the specified data type and shape.

In the following example, the Service expects a NumPy array input with a shape that has any number of rows and 4 columns, with data type ``float32``.
For output, it uses a sample NumPy array (``[[1.0, 2.0, 3.0, 4.0]]``), which means the expected output should resemble a 1x4 matrix of floating-point numbers.

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

For more information, see :ref:`reference/api_io_descriptors:NumPy ``ndarray```.

Pandas DataFrame
~~~~~~~~~~~~~~~~

The ``PandasDataFrame`` IO descriptor is designed to work with Pandas DataFrames, which are commonly used for tabular data.
You specify its data type and shape with the ``dtype`` and ``shape`` arguments. By setting the ``enforce_shape`` and ``enforce_dtype``
arguments to ``True``, the IO descriptor strictly validates the input and output data based on the specified data type and shape.

In the following example, the Service expects a Pandas DataFrame as input, with data type ``float32``. The DataFrame should have any number of rows and 4 columns.
The output descriptor is defined using a sample DataFrame (``[[5,4,3,2]]``), indicating that the expected output should resemble a DataFrame with rows of 4 integer values each.

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

For more information, see :ref:`reference/api_io_descriptors:Tabular Data with Pandas`.

JSON
~~~~

The ``JSON`` IO descriptor is used for handling structured data in JSON format. You can specify its data type through a Pydantic model, which validates the input.

In the following example, the Service uses a Pydantic model that defines the expected structure of the input data. It represents the features of an iris flower,
including sepal length, sepal width, petal length, and petal width, all of which are floating-point numbers. The output is also in JSON format,
specifically a dictionary with a key ``predictions`` that contains the results from the model.

.. code-block:: python

    from typing import Dict, Any
    from pydantic import BaseModel
    from bentoml.io import JSON

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
        input_df = pd.DataFrame([input_series.dict()])
        results = iris_clf_runner.predict.run(input_df).to_list()
        return {"predictions": results}

For more information, see :ref:`reference/api_io_descriptors:Structured Data with JSON` and :examples:`an example project <pydantic_validation>` using Pydantic for request validation.

Composite IO descriptors
^^^^^^^^^^^^^^^^^^^^^^^^

BentoML provides a special IO descriptor called ``Multipart`` to handle cases where you want to accept or produce multiple types of data in a single API call.
It can be used to group multiple IO descriptor instances and each IO descriptor can be customized with independent schema and validation logic.

In the following example, the Service API can accept both a NumPy array and a JSON input in a single call, process them, and then return a NumPy array as a result.
The ``Multipart`` descriptor ensures that these multiple inputs are handled correctly.

.. code-block:: python

    from __future__ import annotations
    from typing import Any
    import numpy as np
    from pydantic import BaseModel

    from bentoml.io import Multipart, NumpyNdarray, JSON

    class IrisFeatures(BaseModel):
        sepal_length: float
        sepal_width: float
        petal_length: float
        petal_width: float

    output_descriptor_numpy = NumpyNdarray.from_sample(np.array([2]))

    @svc.api(
        input=Multipart(
            arr=NumpyNdarray(
                shape=(-1, 4),
                dtype=np.float32,
                enforce_dtype=True,
                enforce_shape=True,
            ),
            json=JSON(pydantic_model=IrisFeatures),
        ),
        output=output_descriptor_numpy,
    )
    def multi_part_predict(arr: np.ndarray, json: dict[str, Any]) -> np.ndarray:
        ...

For more information, see :doc:`/reference/api_io_descriptors`.

Synchronous and asynchronous APIs
---------------------------------

APIs in a BentoML Service can be defined as either synchronous functions or asynchronous coroutines in Python. For synchronous logic,
BentoML creates a pool of workers of optimal size to handle the execution. Synchronous APIs are straightforward and suitable for most of the model serving scenarios.
Here's an example of a synchronous API:

.. code-block:: python

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(input_array: np.ndarray) -> np.ndarray:
        result = runner.run(input_array)
        return result

However, for scenarios where you want to maximize performance and throughput, synchronous APIs may not suffice.
Asynchronous APIs are ideal when the processing logic is IO-bound or invokes multiple Runners simultaneously.
The following asynchronous API example calls a remote feature store asynchronously, invokes two Runners simultaneously, and returns a combined result.

.. code-block:: python

    import aiohttp
    import asyncio

    # Load two Runners for two different versions of the ScikitLearn
    # Iris classifier models saved before
    runner1 = bentoml.sklearn.get("iris_clf:yftvuwkbbbi6zc").to_runner()
    runner2 = bentoml.sklearn.get("iris_clf:edq3adsfhzi6zg").to_runner()

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    async def predict(input_array: np.ndarray) -> np.ndarray:
        # Call a remote feature store to pre-process the request
        async with aiohttp.ClientSession() as session:
            async with session.get('https://features/get', params=input_array[0]) as resp:
                features = get_features(await resp.text())

        # Invoke both model Runners simultaneously
        results = await asyncio.gather(
            runner1.predict.async_run(input_array, features),
            runner2.predict.async_run(input_array, features),
        )
        return combine_results(results)

The asynchronous API implementation is more efficient because when an asynchronous
method is invoked, the event loop becomes available to serve other requests as the current request awaits method results.
In addition, BentoML automatically configures the ideal amount of parallelism based on the available number of CPU cores.
This eliminates the need for further event loop configuration in common use cases.

.. tip::

    Blocking logic such as communicating with an API or database without the ``await``
    keyword will stall the event loop and prevent it from completing other IO tasks.
    If you must use a library that does not support asynchronous IO with ``await``, you
    should use the synchronous API instead. If you are not sure, also use the synchronous
    API to prevent unexpected errors.


.. TODO:

    Running Server:
        bentoml serve arguments
        --reload
        --development

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
