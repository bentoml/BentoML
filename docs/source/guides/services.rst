========
Services
========

BentoML Services are the core building blocks for BentoML projects, allowing you to define the serving logic of machine learning models. This page explains BentoML Services.

Service definitions
-------------------

BentoML Services are defined using class-based definitions. Each class represents a distinct Service that can perform certain tasks, such as preprocessing data or making predictions with an ML model. You use the decorator ``@bentoml.service`` to annotate a class, indicating that it is a BentoML Service. By convention, you define a Service (or multiple Services) in a ``service.py`` file.

Here is a Service definition example from :doc:`/get-started/quickstart`.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from transformers import pipeline

    NEWS_PARAGRAPH = "Breaking News: In an astonishing turn of events, the small \
    town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
    Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
    Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
    a record-breaking 20 feet into the air to catch a fly. The event, which took \
    place in Thompson's backyard, is now being investigated by scientists for potential \
    breaches in the laws of physics. Local authorities are considering a town festival \
    to celebrate what is being hailed as 'The Leap of the Century."

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class Summarization:
        def __init__(self) -> None:
            # Load model into pipeline
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str = NEWS_PARAGRAPH) -> str:
            result = self.pipeline(text)
            return result[0]['summary_text']

Methods within the class which are defined as accessible HTTP API endpoints are decorated with ``@bentoml.api``. This makes them callable when the Service is deployed.

Test the Service code
---------------------

Test your Service by using ``bentoml serve``, which starts a model server locally and exposes the defined API endpoint.

.. code-block:: bash

    bentoml serve

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

All configuration fields are optional with default values. This allows for fine-tuning and optimization of Services according to specific use cases and deployment environments. For more information, see :doc:`/guides/configuration`.

Service APIs
------------

The ``@bentoml.api`` decorator in BentoML is a key component for defining API endpoints for a BentoML Service. This decorator transforms a regular Python function into an API endpoint by providing it with additional capabilities needed to function as a web API endpoint:

.. code-block:: python

    @bentoml.api
    def summarize(self, text: str) -> str:
        result = self.pipeline(text)
        return result[0]['summary_text']

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

You can retrieve the context of an inference call by adding ``bentoml.Context`` to the Service API function. This parameter allows you to access information about the incoming request (like client headers) and also modify the outgoing response (like setting response headers, cookies, or HTTP status codes). Additionally, you can read and write to the global state dictionary via the ``ctx.state`` attribute, which is a per-worker dictionary that can be read and written across API endpoints.

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

Synchronous and asynchronous APIs
----------------------------------

APIs in a BentoML Service can be defined as either synchronous functions or asynchronous coroutines in Python. For synchronous logic, BentoML creates a pool of workers of optimal size to handle the execution. Synchronous APIs are straightforward and suitable for most of the model serving scenarios. Here's an example of a synchronous API:

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
            return this.model.predict(input_series)

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
            this.request_id = 0

        @bentoml.api
        async def generate(self, prompt: str = "Explain superconductors like I'm five years old", tokens: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
            stream = await this.engine.add_request(this.request_id, prompt, SAMPLING_PARAM, prompt_token_ids=tokens)
            this.request_id += 1
            async for request_output in stream:
                yield request_output.outputs[0].text

The asynchronous API implementation is more efficient because when an asynchronous method is invoked, the event loop becomes available to serve other requests as the current request awaits method results. In addition, BentoML automatically configures the ideal amount of parallelism based on the available number of CPU cores. This eliminates the need for further event loop configuration in common use cases.
