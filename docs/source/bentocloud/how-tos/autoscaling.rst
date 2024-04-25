===========
Autoscaling
===========

BentoCloud autoscales your Bento Deployments to efficiently handle varying loads without manual intervention. By dynamically adjusting the number of Service replicas based on incoming traffic and :doc:`concurrency </guides/concurrency>` within the maximum replicas, it ensures optimal resource utilization and cost-efficiency.

This document explains the autoscaling feature of BentoCloud.

Replicas
--------

You can set the :ref:`minimum and maximum replicas <bentocloud/how-tos/configure-deployments:scaling>` to define the boundaries for scaling, allowing the autoscaler to reduce or increase the number of replicas as needed. This feature supports scaling to zero replica during idle time.

Concurrency
-----------

To enable autoscaling, first configure the ``concurrency`` configuration for the service. :doc:`/guides/concurrency` refers to the number of concurrent requests of a BentoML Service is able to to process simultaneously. Setting this parameter means the Service will be automatically scaled on BentoCloud when the concurrent requests per replica exceeds the specified concurrency threshold.

For instance, consider a scenario where ``concurrency`` is set to 32 and the service is currently operating with 2 replicas. If the service receives 100 concurrent requests, BentoCloud will automatically scale up to 4 replicas to effectively manage the increased traffic. Conversely, if the number of concurrent requests decreases to below 32, BentoCloud will intelligently scale down to 1 replica to optimize resource utilization.

In general, the autoscaler will scale the number of replicas based on the following formula, permitted by the ``min_replicas`` and ``max_replicas`` settings in the deployment:

.. image:: ../../_static/img/guides/autoscaling/hpa.png

Use the ``@bentoml.service`` decorator to set concurrency:

.. code-block:: python

    @bentoml.service(
        traffic={
            "concurrency": 32,  # An integer value
        }
    )
    class MyService:
        ...

.. warning::

    If ``concurrency`` is not set, the Service will only be autoscaled based on CPU utilization, which may not be optimal for your Service.

To determine the optimal value for ``concurrency``, we recommend conducting a stress test on your service using a load generation tool such as `Locust <https://locust.io/>`_ either locally or on BentoCloud. The purpose of the stress test is to identify the maximum number of concurrent requests your service can manage. After identifying this maximum, set the concurrency parameter to a value slightly below this threshold ensuring that the service has adequate headroom to handle traffic fluctuations.

External queue
--------------

You can also configure to enable an external request queue to buffer incoming requests. This feature is useful when you want to prevent overloading the Service with requests that exceed the concurrency threshold.

When the external queue is enabled, BentoCloud will hold excess requests in the queue until the Service has the capacity to process them. This ensures that the Service never receives more requests simultaneously than the ``concurrency`` setting allows. BentoCloud will automatically scale the number of replicas based on the number of requests in the queue.

.. note::

    Using the external queue increases the latency of the Service because the extra IO operations are needed to handle the requests in the queue.

You can enhance concurrency management with an external request queue on BentoCloud using the ``@bentoml.service`` decorator:

.. code-block:: python

    @bentoml.service(
        traffic={
            "concurrency": 32,  # An integer value
            "external_queue": True, # A BentoCloud-only field. If set to true, BentoCloud will use an external queue to handle excess requests
        }
    )
    class MyService:
        ...

.. note::

    If you enable ``external_queue`` in the ``@bentoml.service`` decorator, you must specify a ``concurrency`` value.

It's worth noting that when external queue is enabled, ``max_concurrency`` will no longer take effect because BentoCloud guarantees the Service will never receive more requests simultaneously than the ``concurrency`` setting allows. Therefore, ``max_concurrency`` is never reached.

Autoscaling policies
--------------------

You can customize scaling behavior to match your Service's needs with scaling-up and scaling-down policies.

Allowed scaling-up policies (``scale_up_behavior``):

- ``fast`` (default): There is no stabilization window, so the autoscaler can increase the number of replicas immediately if necessary. It can increase the number of replicas by 100% or by 4 replicas, whichever is higher, every 15 seconds.
- ``stable``: The autoscaler can increase the number of replicas, but it will stabilize the number of replicas for 600 seconds (10 minutes) before deciding to scale up further. It can increase the number of replicas by 100% every 15 seconds.
- ``disabled``: Scaling-up is turned off.

Allowed scaling-down policies (``scale_down_behavior``):

- ``fast``: There is no stabilization window, so the autoscaler can reduce the number of replicas immediately if necessary. It can decrease the number of replicas by 100% or by 4 replicas, whichever is higher, every 15 seconds.
- ``stable`` (default): The autoscaler can reduce the number of replicas, but it will stabilize the number of replicas for 600 seconds (10 minutes) before deciding to scale down further. It can decrease the number of replicas by 100% every 15 seconds.
- ``disabled``: Scaling-down is turned off.

To set autoscaling policies, you need to configure the above fields in a separate YAML or JSON file. For example:

.. code-block:: yaml
    :caption: `config-file.yaml`

    services:
      MyBentoService: # The Service name
        scaling:
          max_replicas: 2
          min_replicas: 1
          policy:
            scale_down_behavior: "disabled | stable | fast"  # Choose the behavior
            scale_up_behavior: "disabled | stable | fast"  # Choose the behavior

You can then deploy your project by referencing this file.

.. tab-set::

    .. tab-item:: BentoML CLI

        .. code-block:: bash

            bentoml deploy . -f config-file.yaml

    .. tab-item:: Python API

        .. code-block:: python

            import bentoml
            # Set `bento` to the Bento name if it already exists
            bentoml.deployment.create(bento = "./path_to_your_project", config_file="config-file.yaml")
