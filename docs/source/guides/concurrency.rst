===========
Concurrency
===========

Concurrency refers to the ability of a BentoML Service to process multiple requests simultaneously. It directly impacts the throughput, latency, and overall performance of machine learning models in production. Proper concurrency management ensures that a Service can handle varying loads efficiently, maximizing resource utilization while maintaining high-quality service (QoS).

This document explains how to configure concurrency for BentoML :doc:`/guides/services`.

Configure concurrency
---------------------

To specify concurrency for a BentoML Service, use the concurrency field in traffic within the ``@bentoml.service`` decorator when setting :doc:`configurations </guides/configurations>`:

.. code-block:: python

    @bentoml.service(
        traffic={
            "concurrency": 1,  # An integer value
        }
    )
    class MyService:
        ...

Key points about concurrency in BentoML:

- ``concurrency`` is a new field introduced in BentoML 1.28. It represents the ideal number of requests that a BentoML Service (namely, all :doc:`workers </guides/workers>` in the Service) can simultaneously process. By default, BentoML does not impose a limit on concurrency to avoid bottlenecks.
- If your Service supports :doc:`adaptive batching </guides/adaptive-batching>` or continuous batching, set ``concurrency`` to match the batch size. This aligns processing capacity with batch requirements, optimizing throughput.
- If a Service spawns multiple workers to leverage the parallelism of the underlying hardware accelerators (for example, multi-device GPUs), ``concurrency`` should be configured as the number of parallelism the devices can support.
- For Services designed to handle one request at a time, set ``concurrency`` to ``1``, ensuring that requests are processed sequentially without overlap.

Concurrency and max concurrency
-------------------------------

When using the ``traffic`` field in the ``@bentoml.service`` decorator, you can configure ``concurrency`` and ``max_concurrency`` at the same time, which are both related to how many requests a Service can handle simultaneously.

.. code-block:: python

    @bentoml.service(
        traffic={
            "concurrency": 5,
            "max_concurrency": 10,
        }
    )
    class MyService:
        ...

Note that they serve different purposes:

- ``concurrency``: Indicates the ideal number of simultaneous requests that a Service is designed to handle efficiently. It's a guideline for optimizing performance, particularly in terms of how batching or parallel processing is implemented. This means that the simultaneous requests being processed by a Service instance can still exceed the ``concurrency`` configured.
- ``max_concurrency``: Acts as a hard limit on the number of requests that can be processed simultaneously by a single instance of a Service. It's used to prevent a Service from being overwhelmed by too many requests at once, which could degrade performance or lead to resource exhaustion. Requests that exceed the ``max_concurrency`` limit will be rejected to maintain QoS and ensure that each request is handled within an acceptable time frame. Note that starting from BentoML 1.28, ``max_concurrency`` applies to the aggregate of all workers within a Service. For prior versions, it works on a per-worker basis.

Concurrency-based autoscaling
-----------------------------

For using concurrency-based autoscaling on BentoCloud, see :doc:`/bentocloud/how-tos/autoscaling`.
