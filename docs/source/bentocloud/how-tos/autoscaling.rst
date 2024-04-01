===========
Autoscaling
===========

BentoCloud autoscales your Bento Deployments to efficiently handle varying loads without manual intervention. By dynamically adjusting the number of Service replicas based on incoming traffic and :doc:`concurrency </guides/concurrency>` within the maximum replicas, it ensures optimal resource utilization and cost-efficiency.

This document explains the autoscaling feature of BentoCloud.

Configure autoscaling
---------------------

You can set the :ref:`minimum and maximum values <bentocloud/how-tos/configure-deployments:scaling>` to define the boundaries for scaling, allowing the autoscaler to reduce or increase the number of replicas as needed. This feature supports scaling to zero replica during idle time.

To enable better control over autoscaling, you can set concurrency and external queue.

Concurrency
^^^^^^^^^^^

:doc:`/guides/concurrency` refers to the ability of a BentoML Service to process multiple requests simultaneously. Setting this parameter means the Service will be automatically scaled on BentoCloud when the average requests per replica exceeds the specified concurrency threshold.

Use the ``@bentoml.service`` decorator to set concurrency:

.. code-block:: python

    @bentoml.service(
        traffic={
            "concurrency": 3,  # An integer value
        }
    )
    class MyService:
        ...

External queue
^^^^^^^^^^^^^^

You can enhance concurrency management with an external request queue on BentoCloud using the ``@bentoml.service`` decorator:

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

    - If you enable ``external_queue`` in the ``@bentoml.service`` decorator, you must specify a ``concurrency`` value.
    - ``max_concurrency`` does not take effect on BentoCloud. You need to enable ``external_queue`` to handle excess requests.

Autoscaling policies
--------------------

You can customize scaling behavior to match your Service's needs with scaling-up and scaling-down policies.

Allowed scaling-up policies (``scale_up_behavior``):

- ``fast`` (default): There is no stabilization window, so the autoscaler can increase the number of replicas immediately if necessary. It can increase the number of replicas by 100% or by 4 replicas, whichever is higher, every 15 seconds.
- ``stable``: The autoscaler can increase the number of replicas, but it will stabilize the number of replicas for 300 seconds (5 minutes) before deciding to scale up further. It can increase the number of replicas by 100% every 15 seconds.
- ``disabled``: Scaling-up is turned off.

Allowed scaling-down policies (``scale_down_behavior``):

- ``fast``: There is no stabilization window, so the autoscaler can reduce the number of replicas immediately if necessary. It can decrease the number of replicas by 100% or by 4 replicas, whichever is higher, every 15 seconds.
- ``stable`` (default): The autoscaler can reduce the number of replicas, but it will stabilize the number of replicas for 300 seconds (5 minutes) before deciding to scale down further. It can decrease the number of replicas by 100% every 15 seconds.
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
