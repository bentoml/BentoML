==============
Manage secrets
==============

Secrets can store sensitive data like API keys, authentication tokens, and credentials, essential for accessing various services or resources. BentoCloud provides a secure and flexible environment for injecting required secrets into AI model Deployments.

This document explains how to create and use secrets on BentoCloud.

Create a secret
---------------

From BentoCloud console
~~~~~~~~~~~~~~~~~~~~~~~

1. Navigate to the **Secrets** page and click **Create**.
2. Choose how to create your secret:

   - **From a template**: Use predefined templates for popular services like OpenAI, AWS, Hugging Face, and GitHub. The templates are designed with common keys required for these services.
   - **Custom secrets**: Create a secret by defining custom key-value pairs specific to your model's needs.

3. On the setup page, provide the necessary information for the secret:

   .. image:: ../../_static/img/bentocloud/how-to/manage-secrets/create-a-secret-bentocloud.png

   - **Name**: The name of the secret.
   - **Description**: A description of the secret, detailing its usage.
   - **Mounted as**:

     - **Environment variables**: The environment variables in the secret are accessible in both the image building and Deployment containers.
     - **File**: Inject the secret as a read-only file (or multiple files, depending on the number of key-value pairs in this secret). You need to specify the directory path within the container where the secret will be mounted; ensure this path is under the ``$BENTOML_HOME`` directory.

   - **Key**: The name of the secret key.
   - **Value**: The value of the secret key. If it is mounted as a file, you can optionally specify the sub path where it should be located in the secret path. For example, if the secret path is ``$BENTOML_HOME/demo`` and the sub path is ``secrets/password``, the full path of the secret key will be ``$BENTOML_HOME/demo/secrets/password/<KEY_NAME>``.

4. Click **Add** to add another key-value pair for the secret if necessary.
5. Click **Save** to store the secret.

From BentoML CLI
~~~~~~~~~~~~~~~~

To create a secret from the BentoML command line interface, use the ``secret`` subcommand.

.. code-block:: bash

   bentoml secret create tmy-secret key1=value1 key2=value2


Modify a secret
---------------

1. Click an existing secret on the **Secrets** page.
2. On the details page, click **Edit** and update your desired field.
3. Click **Update** to save your change. It is important to note that:

   - You can't modify a secret key's name.
   - If you change the value of a secret key, you need to restart the Deployment using the secret so that the update can take effect.
   - If you add a new key-value pair or remove any existing key-value pair from the existing secret, we recommend you recreate the Deployment using the secret.

Use secrets for a Deployment
----------------------------

From BentoCloud console
~~~~~~~~~~~~~~~~~~~~~~~

1. During the Deployment creation, select the required secret from the **Secrets** dropdown menu.
2. Attach the desired secret to your Deployment. It will be integrated either as an environment variable or a file, based on the configuration set when the secret was created.

   .. image:: ../../_static/img/bentocloud/how-to/manage-secrets/use-a-secret-for-deployment.png

   .. warning::

      When mounting multiple secrets to a single Deployment, ensure that there are no conflicting key-value pairs across the secrets. For example, different secrets should not contain the same keys with different assigned values.

From BentoML CLI
~~~~~~~~~~~~~~~~

To attach a secret to a deployment, use the ``--secret`` flag when creating a Deployment using the BentoML command line interface.

.. code-block:: bash

   bentoml deploy . --secret my-secret


To attach a secret through a deployment YAML configuration file, add the secret name to the ``secrets`` field.

.. code-block:: yaml

   bento: bentovllm-llama3.1-8b-instruct-service:p34tavtlq25hkasc
   name: bentovllm-llama-3-1-8-b-instruct-service-2qdl
   access_authorization: false
   secrets:
      - my-secret
   services:
      bentovllm-llama3.1-8b-instruct-service:
         instance_type: gpu.l4.1
         envs: []
         scaling:
               min_replicas: 0
               max_replicas: 1
               policy:
                  scale_up_behavior: fast
                  scale_down_behavior: stable
         config_overrides:
               traffic:
                  timeout: 300
                  external_queue: false
                  concurrency: 256
         deployment_strategy: Recreate
   cluster: gcp-us-central-1
