=================
Adaptive batching
=================

Adaptive batching is a dispatching mechanism in BentoML, which adjusts both the batch window and size based on traffic patterns. This mechanism minimizes latency and optimizes resource usage by continuously adjusting the batching parameters based on recent request trends.

.. note::

    Batching means grouping multiple inputs into a single batch for processing. It includes two main concepts:

    - **Batch window**: Maximum time a service waits to accumulate requests into a batch before processing.
    - **Batch size**: Maximum number of requests in a batch.

Architecture
------------

Adaptive batching is implemented on the server side. This is advantageous as opposed to client-side batching because it simplifies the client's logic and it is often times more efficient due to traffic volume.

Specifically, there is a dispatcher within a BentoML Service that oversees collecting requests into a batch until the conditions of the batch window or batch size are met, at which point the batch is sent to the model for inference.

.. image:: ../../_static/img/get-started/adaptive-batching/single-service-batching.png
    :width: 65%
    :align: center
    :alt: Adaptive batching in a single BentoML Service

For multiple Services, the Service responsible for running model inference (``ServiceTwo`` in the diagram below) collects requests from the intermediary Service (``ServiceOne``) and forms batches based on optimal latency.

.. image:: ../../_static/img/get-started/adaptive-batching/multi-service-batching.png
    :width: 100%
    :align: center
    :alt: Adaptive batching in multiple BentoML Services

.. note::

   The ``bentoml.depends()`` function allows one Service to use the functionalities of another. For details, see :doc:`/build-with-bentoml/distributed-services`.

The adaptive batching algorithm continuously learns and adjusts the batching parameters based on recent trends in request patterns and processing time. This means that during high traffic time, batches are likely to be larger and processed more frequently, whereas during quieter periods, BentoML will prioritize reducing latency, even if that means smaller batch sizes.

The order of the requests in a batch is not guaranteed.

Configure adaptive batching
---------------------------

By default, adaptive batching is disabled. Use the ``@bentoml.api`` decorator to enable it and configure the batch behavior for an API endpoint.

Here is an example of enabling batching for the summarization Service in :doc:`hello-world`.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from typing import List
    from transformers import pipeline


    @bentoml.service
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        # Set `batchable` to True to enable batching
        @bentoml.api(batchable=True)
        def summarize(self, texts: List[str]) -> List[str]:
            results = self.pipeline(texts)
            return [item['summary_text'] for item in results]

Note that the batchable API:

- Should be of a type that can encapsulate multiple individual requests, such as ``typing.List[str]`` or ``numpy.ndarray``.
- Only accepts one parameter in addition to ``bentoml.Context``.

You can call the batchable endpoint through a :doc:`BentoML client </build-with-bentoml/clients>`:

.. code-block:: python

    import bentoml
    from typing import List

    client = bentoml.SyncHTTPClient("http://localhost:3000")

    # Specify the texts to summarize
    texts: List[str] = [
        "Paragraph one to summarize",
        "Paragraph two to summarize",
        "Paragraph three to summarize"
    ]

    # Call the exposed API
    response = client.summarize(texts=texts)

    print(f"Summarized results: {response}")

Other available parameters for adaptive batching:

- ``batch_dim``: The batch dimension for both input and output, which can be a tuple or a single value. See :ref:`reference/bentoml/sdk:Service api` for more information.
- ``max_batch_size``: The upper limit for the number of requests that can be grouped into a single batch. Set this parameter based on the available resources, like memory or GPU, to avoid overloading the system.
- ``max_latency_ms``: The maximum time in milliseconds that a batch will wait to accumulate more requests before processing.

When you specify ``max_batch_size`` and ``max_latency_ms`` parameters, BentoML ensures that these constraints are respected, even as it dynamically adjusts batch sizes and processing intervals based on the adaptive batching algorithm. The algorithm's primary goal is to optimize both throughput (by batching requests together) and latency (by ensuring requests are processed within an acceptable time frame). However, it operates within the bounds set by these parameters.

.. note::

    When using a synchronous endpoint in one Service to call a batchable endpoint in another Service, it sends only one request at a time and waits for a response before sending the next. This is due to the default concurrency of 1 for synchronous endpoints. To enable concurrent requests and allow batching, set the ``threads=N`` parameter in the ``@bentoml.service`` decorator.

More BentoML examples with batchable APIs: `SentenceTransformers <https://github.com/bentoml/BentoSentenceTransformers>`_, `CLIP <https://github.com/bentoml/BentoClip>`_ and `ColPali <https://github.com/bentoml/BentoColPali>`_.

Handle multiple parameters
--------------------------

A batchable API endpoint only accepts one parameter in addition to ``bentoml.Context``. For multiple parameters, use a composite input type, such as a :ref:`Pydantic model <build-with-bentoml/iotypes:pydantic>`, to group these parameters into a single object. You also need a wrapper Service to serve as an intermediary to handle individual requests from clients.

Example usage:

.. code-block:: python

    from __future__ import annotations

    from pathlib import Path

    import bentoml
    from pydantic import BaseModel


    # Group together multiple parameters with pydantic
    class BatchInput(BaseModel):
        image: Path
        threshold: float


    # A primary BentoML Service with a batchable API
    @bentoml.service
    class ImageService:
        @bentoml.api(batchable=True)
        def predict(self, inputs: list[BatchInput]) -> list[Path]:
            # Inference logic here using the image and threshold from each input
            # For demonstration, return the image paths directly
            return [input.image for input in inputs]


    # A wrapper Service
    @bentoml.service
    class MyService:
        batch = bentoml.depends(ImageService)

        @bentoml.api
        def generate(self, image: Path, threshold: float) -> Path:
            result = self.batch.predict([BatchInput(image=image, threshold=threshold)])
            return result[0]

In the code snippet:

- The Pydantic model groups together all the required parameters. Each ``BatchInput`` instance represents a single request's parameters, like ``image`` and ``threshold``.
- The primary BentoML Service ``ImageService`` has a batchable API method to accept a list of ``BatchInput`` objects.
- The wrapper Service defines an API ``generate`` that accepts individual parameters (``image`` and ``threshold``) for a single request. It uses ``bentoml.depends`` to invoke the ``ImageService``'s batchable ``predict`` method with a list containing a single ``BatchInput`` instance.

Error handling
--------------

If a Service can't process requests fast enough and exceeds the ``max_latency_ms``, it will return an HTTP 503 Service Unavailable error. To resolve this, either increase ``max_latency_ms`` or improve system resources, such as adding more memory or CPUs.
