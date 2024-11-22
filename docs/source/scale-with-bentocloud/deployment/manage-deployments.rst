==================
Manage Deployments
==================

After you :doc:`deploy a Bento on BentoCloud </get-started/cloud-deployment>`, you can easily manage them using the BentoML CLI or API. Available operations include viewing, updating, applying, terminating, and deleting Deployments.

View
----

To list all Deployments in your BentoCloud account:

.. code-block:: bash

    bentoml deployment list

Expected output:

.. code-block:: bash

    Deployment                                         created_at           Bento                                                                      Status      Region
    sentence-transformers-f8ng                         2024-02-20 17:11:29  sentence_transformers:zf6jipgbyom3denz                                     running     google-cloud-us-central-1
    mistralai-mistral-7-b-instruct-v-0-2-service-cld5  2024-02-20 16:40:16  mistralai--mistral-7b-instruct-v0.2-service:2024-02-03                     running     google-cloud-us-central-1
    summarization                                      2024-02-20 09:27:52  summarization:ghfvclwp2kwm5e56                                             running     aws-ca-1
    control-net-gtb6                                   2024-02-20 01:53:29  control_net:cpvweqwbsgjswpmu                                               terminated  google-cloud-us-central-1
    latent-consistency-4hno                            2024-02-19 03:02:34  latent_consistency:p3ltylgo2kxbwv6m                                        terminated  google-cloud-us-central-1

To retrieve details about a specific Deployment:

.. tab-set::

  .. tab-item:: BentoML CLI

    Choose one of the following commands as needed.

    .. code-block:: bash

      bentoml deployment get <deployment-name>

      # To output the details in JSON
      bentoml deployment get <deployment-name> -o json

      # To output the details in YAML (Default)
      bentoml deployment get <deployment-name> -o yaml

    Expected output in YAML:

    .. code-block:: yaml

        name: summarization
        bento: summarization:ghfvclwp2kwm5e56
        cluster: aws-ca-1
        endpoint_urls:
        - https://summarization-test--aws-ca-1.mt1.bentoml.ai
        admin_console: https://test.cloud.bentoml.com/deployments/summarization/access?cluster=aws-ca-1&namespace=test--aws-ca-1
        created_at: '2024-02-20 09:27:52'
        created_by: bentoml-user
        config:
          envs: []
          services:
            Summarization:
              instance_type: cpu.2
              scaling:
                min_replicas: 1
                max_replicas: 2
              envs: []
              deployment_strategy: Recreate
              extras: {}
              config_overrides:
                traffic:
                  timeout: 10
        status:
          status: running
          created_at: '2024-02-20 09:27:52'
          updated_at: '2024-02-21 05:46:18'

  .. tab-item:: Python API

    To get detailed information about a Deployment:

    .. code-block:: python

      import bentoml

      dep = bentoml.deployment.get(name="deploy-1")
      print(dep.to_dict())  # To output the details in JSON
      print(dep.to_yaml())  # To output the details in YAML

    Expected output in JSON:

    .. code-block:: json

       {
        "name": "deploy-1",
        "bento": "summarization:5vsa3ywqsoefgl7l",
        "cluster": "aws-ca-1",
        "endpoint_urls": [
          "https://deploy-1-test--aws-ca-1.mt1.bentoml.ai"
        ],
        "admin_console": "https://test.cloud.bentoml.com/deployments/deploy-1/access?cluster=aws-ca-1&namespace=test--aws-ca-1",
        "created_at": "2024-03-01 05:00:19",
        "created_by": "bentoml-user",
        "config": {
          "envs": [],
          "services": {
            "Summarization": {
              "instance_type": "cpu.2",
              "scaling": {
                "min_replicas": 1,
                "max_replicas": 1
              },
              "envs": [],
              "deployment_strategy": "Recreate",
              "extras": {},
              "config_overrides": {
                "traffic": {
                  "timeout": 10
                }
              }
            }
          }
        },
        "status": {
          "status": "running",
          "created_at": "2024-03-01 05:00:19",
          "updated_at": "2024-03-06 06:22:53"
         }
       }

    To check the Deployment's status:

    .. code-block:: python

      import bentoml

      dep = bentoml.deployment.get(name="deploy-1")
      status = dep.get_status()
      print(status.to_dict()) # Show the current status of the Deployment
      # Output: {'status': 'running', 'created_at': '2024-03-01 05:00:19', 'updated_at': '2024-03-06 03:55:17'}

    ``get_status()`` has a parameter ``refetch`` to automatically refresh the status, which defaults to ``True``. You can use ``dep.get_status(refetch=False)`` to disable it.

    To get the Deployment's Bento:

    .. code-block:: python

      import bentoml

      dep = bentoml.deployment.get(name="deploy-1")
      bento = dep.get_bento()
      print(bento) # Show the Bento of the Deployment
      # Output: summarization:5vsa3ywqsoefgl7l

    ``get_bento()`` has a parameter ``refetch`` to automatically refresh the Bento information, which defaults to ``True``. You can use ``dep.get_bento(refetch=False)`` to disable it.

    To retrieve configuration details:

    .. code-block:: python

      import bentoml

      dep = bentoml.deployment.get(name="deploy-1")
      config = dep.get_config()
      print(config.to_dict()) # Show the Deployment's configuration details in JSON
      print(config.to_yaml()) # Show the Deployment's configuration details in YAML

    .. note::

       The output is the same as the ``config`` value in the example output above.

    ``get_config()`` has a parameter ``refetch`` to automatically refresh the configuration data, which defaults to ``True``. You can use ``dep.get_config(refetch=False)`` to disable it.

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

You can also update Deployment configurations using a separate file (only add the fields you want to change in the file). This is useful when you have multiple BentoML :doc:`Services </build-with-bentoml/services>` in a Deployment.

.. tab-set::

  .. tab-item:: BentoML CLI

    .. code-block:: bash

      bentoml deployment update <deployment-name> -f patch.yaml

  .. tab-item:: Python API

    .. code-block:: python

      import bentoml

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
