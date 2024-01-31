==========
Deployment
==========

Once you have fully tested your BentoML Service locally, you can push it to BentoCloud with a single command for deployment. BentoCloud is a fully-managed platform designed for running AI applications. It provides comprehensive solutions to deployment, scalability, and collaboration challenges in the AI application delivery lifecycle. As BentoCloud manages the underlying infrastructure for you, you only need to focus on developing AI applications. BentoCloud is currently available with two plans - Starter and Enterprise. See the :doc:`BentoCloud documentation </bentocloud/getting-started/understand-bentocloud>` to learn more.

Deploy a new project to BentoCloud
----------------------------------

The procedures of deploying a new project on BentoCloud can be summarized as follows:

1. Log in to BentoCloud using an :doc:`API token </bentocloud/how-tos/manage-access-token>` with Developer Operations Access.
2. Deploy the project through either the command line interface (CLI) or Python API.

   .. tab-set::

       .. tab-item:: BentoML CLI

           In your project directory where the ``bentofile.yaml`` file is stored, run the following command:

           .. code-block:: bash

              bentoml deploy .

       .. tab-item:: Python API

           Specify the path to your BentoML project using the ``bento`` parameter.

           .. code-block:: python

                import bentoml

                bentoml.deployment.create(bento = "./path_to_your_project")

BentoML does the following automatically during deployment:

1. **Build**: Build your project into a Bento based on ``bentofile.yaml``.
2. **Push**: Push the Bento to BentoCloud.
3. **Deploy**: Deploy the Bento on BentoCloud by performing the following steps in order:

   a. Containerize the Bento as an OCI-compliant image.
   b. Provision instances on BentoCloud.
   c. Start the BentoML Service on the instances based on the specified configuration.

.. note::

   You **DO NOT** need to perform the above three steps (Build, Push, and Deploy) manually, which is a long-running automated process. However, if you want to deploy your BentoML project in environments other than BentoCloud, you can manually build a Bento for your project and deploy it to any Docker-compatible environments. See :doc:`/guides/containerization` for details.

Deploy an existing Bento to BentoCloud
--------------------------------------

If you already have a Bento, either available locally or on BentoCloud, you can use one of the following ways to deploy it.

.. tab-set::

    .. tab-item:: BentoML CLI

        .. code-block:: bash

            bentoml deploy bento_name:version -n <deployment_name>

    .. tab-item:: Python API

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(bento = "bento_name:version", name = "my_deployment_name")

    .. tab-item:: BentoCloud console

        The BentoCloud console provides a web-based, graphical user interface (UI) that you can use to create and manage your Bento Deployments. When you use the BentoCloud console to deploy a Bento, make sure the Bento is already available on BentoCloud.

Customize deployment configurations
-----------------------------------

When deploying a BentoML project on BentoCloud, you can customize the deployment by providing additional configurations to the BentoML CLI command or the Python client.

The following sections provide examples for commonly used configuration fields. See the API reference for details.

.. note::

   You can refer to the following code examples directly if you only have a single BentoML Service in ``service.py``. If it contains multiple Services, see Distributed Services for details.

Scaling
^^^^^^^

You can set the minimum and maximum scaling replicas to ensure efficient resource utilization and cost management.

.. tab-set::

    .. tab-item:: BentoML CLI

        To specify scaling limits via the BentoML CLI, you can use the ``--scaling-min`` and ``--scaling-max`` options.

        .. code-block:: bash

            bentoml deploy . --scaling-min 1 --scaling-max 2

    .. tab-item:: Python API

        When using the Python API, you can specify the scaling limits as arguments in the ``bentoml.deployment.create`` function.

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(
                bento="./path_to_your_project",
                scaling_min=1,
                scaling_max=3
            )

Instance types
^^^^^^^^^^^^^^

You can customize the type of hardware that your Service will run on. This is crucial for performance-intensive applications. If you don’t set an instance type, BentoCloud will automatically infer the most suitable instance based on the ``resources`` field specified in configuration.

To list available instance types on your BentoCloud account, run:

.. code-block:: bash

    $ bentoml deployment list-instance-types

    Name        Price  CPU    Memory  GPU  GPU Type
    cpu.1       *      500m   2Gi
    cpu.2       *      1000m  2Gi
    cpu.4       *      2000m  8Gi
    cpu.8       *      4000m  16Gi
    gpu.t4.1    *      2000m  8Gi     1    nvidia-tesla-t4
    gpu.l4.1    *      4000m  16Gi    1    nvidia-l4
    gpu.a100.1  *      6000m  43Gi    1    nvidia-tesla-a100

.. tab-set::

    .. tab-item:: BentoML CLI

        To set the instance type via the BentoML CLI, use the ``--instance-type`` option followed by the desired instance type name:

        .. code-block:: bash

            bentoml deploy . --instance-type "gpu.a100.1"

    .. tab-item:: Python API

        When using the Python API, you can specify the instance type directly as an argument in the ``bentoml.deployment.create`` function. Here's an example:

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(
                bento="./path_to_your_project",
                instance_type="gpu.a100.1"  # Specify the instance type name here
            )

.. note::

    Choose the instance type that best fits the performance requirements and resource demands of your application. The instance type should be compatible with the deployment environment and supported by the underlying infrastructure.

Environment variables
^^^^^^^^^^^^^^^^^^^^^

You can set environment variables for your deployment to configure the behavior of your BentoML Service, such as API keys, configuration flags, or other runtime settings. During deploying, they will be injected into the image builder container and the Bento Deployment container.

.. tab-set::

    .. tab-item:: BentoML CLI

        To set environment variables via the BentoML CLI, you can use the ``--env`` option:

        .. code-block:: bash

            bentoml deploy . --env AAA=aaa --env BBB=bbb

    .. tab-item:: Python API

        When using the Python API, environment variables are specified through the ``envs`` parameter, which accepts a list of dictionaries. Each dictionary in the list represents a single environment variable. Here's an example:

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(
                bento="./path_to_your_project",
                envs=[
                    {"name": "AAA", "value": "aaa"},  # First environment variable
                    {"name": "BBB", "value": "bbb"}   # Second environment variable
                ]
            )

.. note::

   Ensure that the environment variables you set are relevant to and compatible with your BentoML Service. Use them wisely to manage sensitive data, configuration settings, and other critical information.

If you have multiple Services, you can set environment variables at different levels. For example, setting global environment variables means they will be applied to all Services, while a single Service can have environment variables only specific to itself, which take precedence over global ones. See Distributed Services to learn more.

Deploy with a configuration file
--------------------------------

If you have many custom configuration fields or multiple Services, you can define them in a separate file (YAML or JSON), and reference it in the BentoML CLI or the ``bentoml.deployment.create`` API.

.. tab-set::

    .. tab-item:: BentoML CLI

        .. code-block:: bash

            bentoml deploy . -f config-file.yaml

    .. tab-item:: Python API

        .. code-block:: python

            import bentoml
            bentoml.deployment.create(bento = "./path_to_your_project", config_file="config-file.yaml")

Override configurations
-----------------------

When defining a BentoML Service, you can use the ``@bentoml.service`` decorator to add configurations, such as timeout and resources. These configurations will be applied when you deploy the Service on BentoCloud. However, BentoML also allows you to override these configurations at the time of deployment using the ``config_overrides`` field in the deployment configuration. This provides a flexible way to adapt your Service for different deployment scenarios without changing the Service code.

Suppose you have a BentoML Service defined with certain resource and timeout configurations:

.. code-block:: python

    @bentoml.service(
        resources={"memory": "500MiB"},
        traffic={"timeout": 60},
    )
    class MyBentoService:
        # Service implementation

To override a field (for example, ``timeout``), you need to set it in a separate YAML (or JSON) file and then reference it when deploying the Service. Your YAML file may look like this:

.. code-block:: yaml
    :caption: `config-file.yaml`

    services:
      MyBentoService: # The Service name
        config_overrides:
          traffic:
            timeout: 30 # Change the timeout from 60 seconds to 30 seconds

You can then deploy your project by referencing this file.

.. note::

   - Always ensure that the overrides are consistent with the capabilities of the deployment environment (for example, available resources on the cluster).
   - It is important to thoroughly test these configurations to ensure that the Service operates as expected.

Deployment strategies
---------------------

BentoML supports various deployment strategies, allowing you to choose how updates to your Service are rolled out. The choice of strategy can impact the availability, speed, and risk level of deployments.

Available strategies include:

- ``RollingUpdate``: Gradually replaces the old version with the new version. This strategy minimizes downtime but can temporarily mix versions during the rollout.
- ``Recreate``: All existing replicas are killed before new ones are created. This strategy can lead to downtime but it is fast and ensures that only one version of the application is running at a time. ``Recreate`` is the default rollout strategy. You can update it to use another one after deploying your application.
- ``RampedSlowRollout``: Similar to ``RollingUpdate``, but with more control over the speed of the rollout. It's useful for slowly introducing changes and monitoring their impact.
- ``BestEffortControlledRollout``: Attempts to minimize the risk by gradually rolling out changes, but adapts the rollout speed based on the success of the deployment.

.. tab-set::

    .. tab-item:: BentoML CLI

        To set a deployment strategy via the BentoML CLI, you can use the ``--strategy`` option:

        .. code-block:: bash

            bentoml deploy . --strategy Recreate

    .. tab-item:: Python API

        To set a deployment strategy using the Python API, you can specify it directly as an argument in the ``bentoml.deployment.create`` function. Here's an example:

        .. code-block:: bash

            import bentoml

            bentoml.deployment.create(
                bento="./path_to_your_project",
                strategy="RollingUpdate"  # Specify the deployment strategy here
            )

Autoscaling policies
--------------------

The autoscaling feature dynamically adjusts the number of Pods within the specified minimum and maximum limits. You can define the minimum and maximum values to define the boundaries for scaling, allowing the autoscaler to reduce or increase the number of Pods as needed. This feature supports scaling to zero Pods. You can also define the specific metric thresholds that the autoscaler will use to determine when to adjust the number of Pods. The available ``metrics`` values include:

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
