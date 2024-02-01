==================
Manage Deployments
==================

After you :doc:`deploy a Bento on BentoCloud </guides/deployment>`, you can easily manage them using the BentoML CLI or API. Available operations include viewing, updating, applying, terminating, and deleting Deployments.

Get
---

To retrieve details about a specific Deployment:

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      bentoml deployment get <deployment-name>

  .. tab-item:: Python API

    .. code-block:: python

      bentoml.deployment.get(name="deployment-1")

Update
------

Updating a Deployment is essentially a patch operation. This means that when you execute an update command, it only modifies the specific fields that are explicitly included in the update command. All other existing fields and configurations of the Deployment remain unchanged. This is useful for making incremental changes to a Deployment without needing to redefine the entire configuration.

To update specific parameters of a single-Service Deployment:

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      # Add the parameter name flag
      bentoml deployment update <deployment-name> --scaling-min 1
      bentoml deployment update <deployment-name> --scaling-max 5

  .. tab-item:: Python API

    .. code-block:: python

      import bentoml

      bentoml.deployment.update(
        name = "deployment-1",
        scaling_min=1,
        scaling_max=3
        # No change to unspecified parameters
      )

You can also update Deployment configurations using a separate file (only add the fields you want to change in the file). This is useful when you have multiple BentoML :doc:`/guides/services` in a Deployment.

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      bentoml deployment update <deployment-name> -f patch.yaml

  .. tab-item:: Python API

    .. code-block:: python

      bentoml.deployment.update(name="deployment-1", config_file="patch.yaml")

To roll out a Deployment:

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      # Use the Bento name
      bentoml deployment update <deployment-name> --bento bento_name:version

      # Use the project directory
      bentoml deployment update <deployment-name> --bento ./project/directory

  .. tab-item:: Python API

    .. code-block:: python

      import bentoml

      # Use the Bento name
      bentoml.deployment.update(name="deployment-1", bento="bento_name:version")

      # Use the project directory
      bentoml.deployment.update(name="deployment-1", project_path="./project/directory")

Apply
-----

The ``apply`` operation is a comprehensive way to manage Deployments, allowing you to create or update a Deployment based on the specifications provided. It works in the following ways:

- If a Deployment with the given name does not exist, ``apply`` will create a new Deployment based on the specifications provided.
- If a Deployment with the specified name already exists, ``apply`` will update the existing Deployment to match the provided specifications exactly.

The differences between ``apply`` and ``update``:

- **Update (Patch-only):** Makes minimal changes, only updating what you specify.
- **Apply (Overriding):** Considers the entire configuration and may reset unspecified fields to their default values or remove them if they're not present in the applied configuration. If a Deployment does exist, applying the configuration will create the Deployment.

To apply new configurations to a Deployment, you define them in a separate file as reference.

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      bentoml deployment apply <deployment_name> -f new_deployment.yaml

  .. tab-item:: Python API

    .. code-block:: python

      import bentoml

      bentoml.deployment.apply(name = "deployment-1", config_file = "deployment.yaml")

Terminate
---------

Terminating a Deployment means it will be stopped so that it does not incur any cost. You can still restore a Deployment after it is terminated.

To terminate a Deployment:

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      bentoml deployment terminate <deployment_name>

  .. tab-item:: Python API

    .. code-block:: python

      import bentoml
      bentoml.deployment.terminate(name="deployment-1")

Delete
------

You can delete a Deployment if you no longer need it. To delete a Deployment:

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      bentoml deployment delete <deployment_name>

  .. tab-item:: Python API

    .. code-block:: python

      import bentoml
      bentoml.deployment.delete(name="deployment-1")

.. warning::

    Exercise caution when deleting a Deployment. This action is irreversible.
