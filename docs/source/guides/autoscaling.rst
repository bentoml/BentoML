===========
Autoscaling
===========

Concurrency refers to the ability of a BentoML Service to process multiple requests simultaneously. It directly impacts the throughput, latency, and overall performance of machine learning models in production. Proper concurrency management ensures that a Service can handle varying loads efficiently, maximizing resource utilization while maintaining high-quality service (QoS).

This document explains how to configure concurrency for BentoML Services.

Configure concurrency
---------------------

To specify concurrency for a BentoML Service, use the concurrency field in traffic within the ``@bentoml.service`` decorator when setting configurations:

.. code-block:: python

    @bentoml.service(
        traffic={
            "concurrency": 1,  # An integer value
        }
    )
    class MyService:
        ...

Key points about concurrency in BentoML:

- Concurrency represents the ideal number of requests a Service can simultaneously process. By default, BentoML does not impose a limit on concurrency to avoid bottlenecks.
- If your Service supports adaptive batching or continuous batching, set ``concurrency`` to match the batch size. This aligns processing capacity with batch requirements, optimizing throughput.
- If a Service spawns multiple workers to leverage the parallelism of the underlying hardware accelerators (for example, multi-device GPUs), ``concurrency`` should be configured as the number of parallelism the devices can support.
- For Services designed to handle one request at a time, set ``concurrency`` to ``1``, ensuring that requests are processed sequentially without overlap.

Concurrency and max concurrency
-------------------------------

When using the ``traffic`` field in the ```@bentoml.service``` decorator, you can configure ``concurrency`` and ``max_concurrency`` at the same time, which are both related to how many requests a Service can handle simultaneously. However, they serve different purposes and should be used 

- ``concurrency``: Indicates the ideal number of simultaneous requests that a Service is designed to handle efficiently. It's a guideline for optimizing performance, particularly in terms of how batching or parallel processing is implemented. Note that the simultaneous requests being processed by a Service instance can still exceed the ``concurrency`` configured.
- ``max_concurrency``: Acts as a hard limit on the number of requests that can be processed simultaneously by a single instance of a Service. It's used to prevent a Service from being overwhelmed by too many requests at once, which could degrade performance or lead to resource exhaustion. Requests that exceed the ``max_concurrency`` limit will be rejected to maintain QoS and ensure that each request is handled within an acceptable time frame.

How does concurrency work on BentoCloud
---------------------------------------

Autoscaling
^^^^^^^^^^^

When a Service is deployed to BentoCloud, the serverless platform dynamically adjusts the number of Service replicas based on the incoming traffic and the concurrency value. Autoscaling ensures that your Service can handle varying loads efficiently without exceeding the maximum replicas configured.

External queue
^^^^^^^^^^^^^^

You can enhance concurrency management with an external request queue on BentoCloud:

.. code-block:: python

    @bentoml.service(
    traffic={
        "concurrency": 3,  # An integer value
        "external_queue": True, # A BentoCloud-only field. If set to true, BentoCloud will use an external queue to handle excess requests
    }
    )
    class MyService:
        ...


The external request queue is used to moderate incoming traffic, ensuring that a Service instance never receives more requests simultaneously than the ``concurrency`` setting allows. Excess requests are held in the queue until the Service has the capacity to process them, preventing overloading and maintaining efficient operation.


.. note::

    1. If you enable `external_queue` in the `@bentoml.service` decorator, you must specify a `concurrency` value.
    2. `max_concurrency` does not take effect on BentoCloud. You need to enable `external_queue` to handle excess requests.

The following content will be updated to the `traffic` section of the existing :doc:`Configurations </guides/configurations>` doc.

``traffic``
^^^^^^^^^^^

``traffic`` in BentoML allows you to manage how requests are handled by your Service. It includes settings for managing request concurrency and ensuring timely responses, which are helpful for optimizing the Service's responsiveness and load management. The following fields are available:

- ``timeout``: Determines the maximum time the Service will wait for a response to be sent back to the client. The default timeout is set to 60 seconds.
- ``concurrency``: Represents the ideal number of simultaneous requests a Service is designed to handle. Concurrency helps optimize resource utilization and influences how BentoCloud autoscales your Service. By default, concurrency is set to allow infinite requests to avoid system bottlenecks.
- ``max_concurrency``: Specifies the hard limit on the number of requests that can be processed simultaneously by a single Service instance. It helps you control the load and prevent the Service from being overwhelmed by too many simultaneous requests.
- ``external_queue``: A BentoCloud-specific field. When deploying a Service on BentoCloud with this field enabled, an external request queue is used to manage incoming traffic more effectively. This is done by queuing excess requests until they can be processed within the defined concurrency limits.
