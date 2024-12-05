==========================
Create online API Services
==========================

BentoML Services are the core building blocks for BentoML projects, allowing you to define the serving logic of machine learning models. This page explains BentoML Services.

Service definitions
-------------------

BentoML Services are defined using class-based definitions. Each class represents a distinct Service that can perform certain tasks, such as preprocessing data or making predictions with an ML model. You use the decorator ``@bentoml.service`` to annotate a class, indicating that it is a BentoML Service. By convention, you define a Service (or multiple Services) in a ``service.py`` file. For more information, see :ref:`reference/bentoml/sdk:Service decorator`.

Here is a Service definition example from :doc:`/get-started/hello-world`.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from transformers import pipeline

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class Summarization:
        def __init__(self) -> None:
            # Load model into pipeline
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str) -> str:
            result = self.pipeline(text)
            return result[0]['summary_text']

Methods within the class which are defined as accessible HTTP API endpoints are decorated with ``@bentoml.api``. This makes them callable when the Service is deployed.

.. note::

    This Service downloads a pre-trained model from Hugging Face. It is possible to use your own model within the Service class. For more information, see :doc:`/build-with-bentoml/model-loading-and-management`.

Test the Service code
---------------------

Test your Service by using ``bentoml serve``, which starts a model server locally and exposes the defined API endpoint.

.. code-block:: bash

    bentoml serve <service:class_name>

By default, the server is accessible at `http://localhost:3000/ <http://localhost:3000/>`_. Specifically, ``bentoml serve`` does the following:

- Turns API code into a REST API endpoint. ``POST`` is the default HTTP method.
- Manages the lifecycle of the Service instance defined.
- Creates a URL route based on the method name. In this example, it is ``http://localhost:3000/summarize``. This route can be customized.

Service configurations
----------------------

You use the ``@bentoml.service`` decorator to specify Service-level configurations like resource requirements and timeout. These configurations are applied both when you serve the Service locally and deploy the resulting Bento on BentoCloud (or the Bento image as containers in environments like Kubernetes). For example:

.. code-block:: python

    @bentoml.service(
        resources={"memory": "500MiB"},
        traffic={"timeout": 10},
    )
    class Summarization:
        # Service definition here

All configuration fields are optional with default values. This allows for fine-tuning and optimization of Services according to specific use cases and deployment environments.

Service APIs
------------

The ``@bentoml.api`` decorator in BentoML is a key component for defining API endpoints for a BentoML Service. This decorator transforms a regular Python function into an API endpoint by providing it with additional capabilities needed to function as a web API endpoint:

.. code-block:: python

    @bentoml.api
    def summarize(self, text: str) -> str:
        result = self.pipeline(text)
        return result[0]['summary_text']

You can customize the input and output logic of the Service API. See :doc:`/build-with-bentoml/iotypes` to learn more.

Custom route path
^^^^^^^^^^^^^^^^^

Each API endpoint has a unique route (URL path). By default, the route is derived from the function name, but you can customize it using the ``route`` parameter.

.. code-block:: python

    @bentoml.api(route="/custom/url/name")
    def summarize(self, text: str) -> str:
        result = self.pipeline(text)
        return result[0]['summary_text']

Inference context
^^^^^^^^^^^^^^^^^

You can retrieve the context of an inference call by adding ``bentoml.Context`` to the Service API function. This parameter allows you to access information about the incoming request (like client headers) and also modify the outgoing response (like setting response headers, cookies, or HTTP status codes). Additionally, you can read and write to the global state dictionary via the ``ctx.state`` attribute, which is a :doc:`per-worker </build-with-bentoml/parallelize-requests>` dictionary that can be read and written across API endpoints.

.. code-block:: python

    @bentoml.api
    def summarize(self, text: str, ctx: bentoml.Context) -> str:
        # Get request headers
        request_headers = ctx.request.headers

        result = self.pipeline(text)

        # Set response headers, cookies, and status code
        ctx.response.status_code = 202
        ctx.response.cookies = [
            bentoml.Cookie(
                key="key",
                value="value",
                max_age=None,
                expires=None,
                path="/summarize",
                domain=None,
                secure=True,
                httponly=True,
                samesite="None"
            )
        ]

        # Add a custom header to the response
        ctx.response.headers.append("X-Custom-Header", "value")

        return result[0]['summary_text']

Lifecycle hooks
^^^^^^^^^^^^^^^

BentoML's lifecycle hooks provide a way to insert custom logic at specific stages of a Service's lifecycle.

- Deployment hooks (``@bentoml.on_deployment``): Execute global setup actions before :doc:`Service workers </build-with-bentoml/parallelize-requests>` are spawned. They run only once regardless of the number of workers, ideal for one-time initializations.
- Shutdown hooks (``@bentoml.on_shutdown``): Run cleanup logic when a BentoML Service is shutting down. They enable tasks such as closing connections and releasing resources to ensure a graceful shutdown.

You use decorators to set lifecycle hooks. For details, see :doc:`/build-with-bentoml/lifecycle-hooks`.

Synchronous and asynchronous APIs
---------------------------------

APIs in a BentoML Service can be defined as either synchronous functions or asynchronous coroutines in Python.

Basic usage
^^^^^^^^^^^

For synchronous logic, BentoML creates a pool of workers of optimal size to handle the execution. Synchronous APIs are straightforward and suitable for most of the model serving scenarios. Here's an example of a synchronous API:

.. code-block:: python
   :emphasize-lines: 11, 12, 13

    @bentoml.service(name="iris_classifier", resources={"cpu": "200m", "memory": "512Mi"})
    class IrisClassifier:
        iris_model = bentoml.models.get("iris_sklearn:latest")
        preprocessing = bentoml.depends(Preprocessing)

        def __init__(self):
            import joblib

            self.model = joblib.load(self.iris_model.path_of("model.pkl"))

        @bentoml.api
        def classify(self, input_series: np.ndarray) -> np.ndarray:
            return self.model.predict(input_series)

However, for scenarios where you want to maximize performance and throughput, synchronous APIs may not suffice. Asynchronous APIs are ideal when the processing logic is IO-bound and async model execution is supported. Here is an example:

.. code-block:: python
   :emphasize-lines: 15, 16, 17, 18, 19, 20

    import bentoml

    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from typing import Optional, AsyncGenerator, List

    SAMPLING_PARAM = SamplingParams(max_tokens=4096)
    ENGINE_ARGS = AsyncEngineArgs(model='meta-llama/Llama-2-7b-chat-hf')

    @bentoml.service(workers=1, resources={"gpu": "1"})
    class VLLMService:
        def __init__(self) -> None:
            self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
            self.request_id = 0

        @bentoml.api
        async def generate(self, prompt: str = "Explain superconductors like I'm five years old", tokens: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
            stream = await self.engine.add_request(self.request_id, prompt, SAMPLING_PARAM, prompt_token_ids=tokens)
            self.request_id += 1
            async for request_output in stream:
                yield request_output.outputs[0].text

The asynchronous API implementation is more efficient because when an asynchronous method is invoked, the event loop becomes available to serve other requests as the current request awaits method results. In addition, BentoML automatically configures the ideal amount of parallelism based on the available number of CPU cores. This eliminates the need for further event loop configuration in common use cases.

.. warning::

    Avoid implementating blocking logic within asynchronous APIs, since such operations can block the IO event loop, preventing health check endpoints like ``/readyz`` from functioning properly.

Convert synchronous to asynchronous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For machine learning inference tasks, though traditionally executed synchronously, may require asynchronous execution for different reasons, such as:

- Running tasks in parallel
- Using resources like databases that support asynchronous connections

However, directly calling synchronous blocking functions within an asynchronous context is generally considered bad practice, as this can block the event loop, leading to decreased performance and responsiveness. In such cases, you can use the ``.to_async`` property of a Service, which allows you to convert synchronous methods of the Service to an asynchronous one. This can enable non-blocking execution and improve performance in IO-bound operations. Here is an example:

.. code-block:: python
   :emphasize-lines: 29, 30

    ...
    @bentoml.service(
        traffic={"timeout": 600},
        workers=4,
        resources={
            "memory": "4Gi"
        },
    )
    class GreetingCardService:
        # Services StableLMService, SDXLTurboService, and XTTSService are previously defined
        # Retrieve these Services using `bentoml.depends` so that their methods can be called directly
        stablelm = bentoml.depends(StableLMService)
        sdxl = bentoml.depends(SDXLTurboService)
        xtts = bentoml.depends(XTTSService)

        @bentoml.api
        async def generate_card(
                self,
                context: bentoml.Context,
                message: str = "Happy new year!",
        ) -> Annotated[Path, bentoml.validators.ContentType("video/*")]:
            greeting_message = await self.stablelm.enhance_message(message)

            sdxl_prompt_tmpl = "a happy and heart-warming greeting card based on greeting message {message}"
            sdxl_prompt = sdxl_prompt_tmpl.format(message=greeting_message)

            # Run `txt2img` and `synthesize` operations in parallel
            audio_path, image = await asyncio.gather(
                self.xtts.to_async.synthesize(greeting_message),
                self.sdxl.to_async.txt2img(sdxl_prompt)
            )

            image_path = os.path.join(context.temp_dir, "output.png")
            image.save(image_path)

            cmd = ["ffmpeg", "-loop", "1", "-i", str(image_path), "-i", str(audio_path), "-shortest"]
            output_path = os.path.join(context.temp_dir, "output.mp4")
            cmd.append(output_path)
            subprocess.run(cmd)

            return Path(output_path)

.. note::

    ``bentoml.depends()`` is commonly used for interservice communication as it allows you to directly call the API methods of a BentoML Service within another Service as if they were local class functions. For more information, see :doc:`/build-with-bentoml/distributed-services`.

In this example, the ``.to_async`` property converts synchronous methods (``txt2img`` and ``synthesize`` of ``SDXLTurboService`` and ``XTTSService`` respectively) into their asynchronous versions, enabling the ``generate_card`` method to perform multiple asynchronous operations concurrently with ``asyncio.gather``.

.. _bentoml-tasks:

Tasks
-----

Tasks in BentoML allow you to execute long-running operations in the background, managed via a task queue style API. These background tasks are ideal for scenarios like batch processing and image or video generation where you don't need the results immediately or synchronously.

To define a task endpoint, use the ``@bentoml.task`` decorator in the Service constructor. For more information, see :doc:`/get-started/async-task-queues`.

Convert legacy Runners to a Service
-----------------------------------

`Runners <https://docs.bentoml.com/en/v1.1.11/concepts/runner.html>`_ are a legacy concept in BentoML 1.1, which represent a computation unit that can be executed on a remote Python worker and scales independently. In BentoML 1.1, Services are defined using both ``Service`` and ``Runner`` components, where a Service could contain one or more Runners. Starting with BentoML 1.2, the framework has been streamlined to use a Python class to define a BentoML Service.

To minimize code changes when migrating from 1.1 to 1.2+, you can use the ``bentoml.runner_service()`` function to convert Runners to a Service. Here is an example:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import numpy as np


    # Create a legacy runner
    sample_legacy_runner = bentoml.models.get("model_name:version").to_runner()
    # Create an internal Service
    SampleService = bentoml.runner_service(runner = sample_legacy_runner)

    # Use the @bentoml.service decorator to mark a class as a Service
    @bentoml.service(
        resources={"cpu": "2", "memory": "500MiB"},
        workers=1,
        traffic={"timeout": 20},
    )
    # Define the BentoML Service
    class MyService:
        # Integrate the internal Service using bentoml.depends() to inject it as a dependency
        sample_model_runner = bentoml.depends(SampleService)

        # Define Service API and IO schema
        @bentoml.api
        def classify(self, input_series: np.ndarray) -> np.ndarray:
            # Use the internal Service for prediction
            result = self.sample_model_runner.predict.run(input_series)
            return result
