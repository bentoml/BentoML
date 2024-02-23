===========
Autoscaling
===========

The autoscaling feature of BentoCloud dynamically adjusts the number of Service replicas within the specified minimum and maximum limits. This document explains how to set autoscaling for Deployments.

You can define the minimum and maximum values to define the boundaries for scaling, allowing the autoscaler to reduce or increase the number of replicas as needed. This feature supports scaling to zero replica. You can also define the specific metric thresholds that the autoscaler will use to determine when to adjust the number of replicas. The available ``metrics`` values include:

- ``cpu``: The CPU utilization percentage.
- ``memory``: The memory utilization.
- ``gpu``: The GPU utilization percentage.
- ``qps``: The queries per second.

By setting values for these fields, you are instructing the autoscaler to ensure that the average for each metric does not exceed the specified thresholds. For example, if you set the CPU value to ``80``, the autoscaler will target an average CPU utilization of 80%.

Allowed scaling-up behaviors (``scale_up_behavior``):

- ``fast`` (default): There is no stabilization window, so the autoscaler can increase the number of Pods immediately if necessary. It can increase the number of Pods by 100% or by 4 Pods, whichever is higher, every 15 seconds.
- ``stable``: The autoscaler can increase the number of Pods, but it will stabilize the number of Pods for 300 seconds (5 minutes) before deciding to scale up further. It can increase the number of Pods by 100% every 15 seconds.
- ``disabled``: Scaling-up is turned off.

Allowed scaling-down behaviors (``scale_down_behavior``):

- ``fast``: There is no stabilization window, so the autoscaler can reduce the number of Pods immediately if necessary. It can decrease the number of Pods by 100% or by 4 Pods, whichever is higher, every 15 seconds.
- ``stable`` (default): The autoscaler can reduce the number of Pods, but it will stabilize the number of Pods for 300 seconds (5 minutes) before deciding to scale down further. It can decrease the number of Pods by 100% every 15 seconds.
- ``disabled``: Scaling-down is turned off.

To set autoscaling, you need to configure the above fields in a separate YAML or JSON file. For example:

.. code-block:: yaml
    :caption: `config-file.yaml`

    services:
      MyBentoService: # The Service name
        scaling:
          max_replicas: 2
          min_replicas: 1
          policy:
            metrics:
              - type: "cpu | memory | gpu | qps"  # Specify the type here
                value: "string"  # Specify the value here
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
