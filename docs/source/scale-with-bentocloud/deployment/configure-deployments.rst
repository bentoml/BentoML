=====================
Configure Deployments
=====================

When deploying a BentoML project on BentoCloud, you can customize the deployment by providing additional configurations to the BentoML CLI command or the Python client.

This document provide examples for setting commonly used configurations.

Configurations
--------------

Refer to the following code examples directly if you only have a single BentoML Service in ``service.py``. If it contains multiple Services, see :ref:`deploy-with-config-file` and :doc:`/build-with-bentoml/distributed-services` for details.

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

For more information, see :doc:`/scale-with-bentocloud/scaling/autoscaling`.

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

.. important::

    You DO NOT need to set the same environment variables again if you have already specified them in ``bentofile.yaml`` using the ``envs`` field. See :doc:`/reference/bentoml/bento-build-options` for details.

.. tab-set::

    .. tab-item:: BentoML CLI

        To set environment variables via the BentoML CLI, you can use the ``--env`` option:

        .. code-block:: bash

            bentoml deploy . --env FIRST_VAR_NAME=value --env SECOND_VAR_NAME=value

    .. tab-item:: Python API

        When using the Python API, environment variables are specified through the ``envs`` parameter, which accepts a list of dictionaries. Each dictionary in the list represents a single environment variable. Here's an example:

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(
                bento="./path_to_your_project",
                envs=[
                    {"name": "FIRST_VAR_NAME", "value": "first_var_value"},  # First environment variable
                    {"name": "SECOND_VAR_NAME", "value": "second_var_value"}   # Second environment variable
                ]
            )

.. note::

   Ensure that the environment variables you set are relevant to and compatible with your BentoML Service. Use them wisely to manage sensitive data, configuration settings, and other critical information.

If you have multiple Services, you can set environment variables at different levels. For example, setting global environment variables means they will be applied to all Services, while a single Service can have environment variables only specific to itself, which take precedence over global ones. See :doc:`/build-with-bentoml/distributed-services` to learn more.

Authorization
^^^^^^^^^^^^^

Enabling authorization for a Deployment in BentoCloud is essential for security reasons. It allows you to control access to a Deployment by creating a protected endpoint, ensuring that only individuals with a valid token can access it. This mechanism helps in safeguarding sensitive data and functionality exposed by the application, preventing unauthorized access and potential misuse.

.. tab-set::

    .. tab-item:: BentoML CLI

        To set authorization via the BentoML CLI, you can use the ``--access-authorization`` option:

        .. code-block:: bash

            bentoml deploy . --access-authorization true

    .. tab-item:: Python API

        Set the ``access_authorization`` parameter to ``True`` to enable it.

        .. code-block:: python

            import bentoml

            bentoml.deployment.create(
                bento="./path_to_your_project",
                access_authorization=True
            )

To access a Deployment with authorization enabled, :ref:`create an API token with Protected Endpoint Access <scale-with-bentocloud/manage-api-tokens:create an api token>` and refer to :ref:`scale-with-bentocloud/manage-api-tokens:access protected deployments`.

.. _deploy-with-config-file:

Deploy with a configuration file
--------------------------------

If you have many custom configuration fields or multiple Services, you can define them in a separate file (YAML or JSON), and reference it in the BentoML CLI or the ``bentoml.deployment.create`` API.

Here is an example ``config-file.yaml`` file:

.. code-block:: yaml
    :caption: `config-file.yaml`

    name: "my-deployment-name"
    bento: .
    access_authorization: true # Setting it to `true` means you need an API token with Protected Endpoint Access to access the exposed endpoint.
    envs: # Set global environment variables
      - name: ENV_VAR_NAME
        value: env_var_value
    services:
        MyBentoService: # Your Service name
          instance_type: "cpu.2" # The instance type name on BentoCloud
          scaling: # Set the max and min replicas for scaling
            min_replicas: 1
            max_replicas: 3
          deployment_strategy: "Recreate"
        # Add another Service below if you have more

You can then create a Deployment as below:

.. tab-set::

    .. tab-item:: BentoML CLI

        .. code-block:: bash

            bentoml deploy -f config-file.yaml

    .. tab-item:: Python API

        .. code-block:: python

            import bentoml
            bentoml.deployment.create(config_file="config-file.yaml")

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

See also
--------

- :doc:`/scale-with-bentocloud/deployment/manage-deployments`
- :doc:`/reference/bentocloud/bentocloud-cli`
- :doc:`/reference/bentocloud/bentocloud-api`
