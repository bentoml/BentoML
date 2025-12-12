=================
Manage API tokens
=================

In BentoCloud, API tokens serve as a key method of authorization. You may use tokens to:

- Log in to BentoCloud
- Manage BentoCloud resources
- Access protected Deployments, which have :ref:`scale-with-bentocloud/deployment/configure-deployments:Authorization` enabled

This tutorial explains how to create and use API tokens in BentoCloud.

Types of API tokens
-------------------

BentoCloud offers two types of API tokens:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Personal API token
     - Organization API token
   * - Ownership
     - Belongs to the individual user
     - Belongs to the organization
   * - Management
     - Can be managed by the token creator
     - Visible to all members, but only admins can create, update or delete tokens, and view values
   * - Validity
     - Becomes invalid if the creator leaves the organization
     - Remains valid even if the creator leaves the organization. Other admins can still manage it
   * - Use cases
     - Individual development and testing
     - Continuous integration/deployment (CI/CD) pipelines, regular automated tasks, shared and long-term workflows

For ongoing automated tasks requiring frequent updates, deployments, or access to protected endpoints, we recommend Organization API tokens to ensure continuity. Personal API Tokens become invalid when their associated user leaves the organization.

.. _creating-an-api-token:

Create an API token
-------------------

1. Click your profile image in the top-right corner of any BentoCloud page, then select **API Tokens**.
2. Choose either Personal API Tokens or Organization API Tokens, and click **Create**.
3. In the dialog that appears, specify the following fields. Note that you must select at least one of the access types.

   .. image:: ../../_static/img/bentocloud/how-to/manage-access-tokens/token-creation-dialog.png
      :alt: Screenshot of BentoCloud API token creation dialog showing fields for name, description, access permissions, and expiration date

   - **Name**: The name of the API token.
   - **Description**: A description of the token, detailing its usage.
   - **Developer Operations Access**: Grant permissions to access BentoCloud and manage resources on it.
   - **Protected Endpoint Access**: Grant permissions to access Bento Deployments with Protected endpoints. If you select this type, you need to choose the Deployment that you want the token to access. If you want to use the token to access all the Protected Deployments, select **All Deployments**.
   - **Expired At**: Set an expiration date for the token. You won't be able to use the token after it expires.

4. Click **Submit**.
5. Record the token.
6. All available tokens appear on the **API Tokens** page. Click **Delete** if you no longer need a token.

Log in to BentoCloud using the BentoML CLI
------------------------------------------

CLI login requires an API token with Developer Operations Access.

1. Run the ``bentoml cloud login`` command.

   .. code-block:: bash

      bentoml cloud login

2. Follow the on-screen instructions to log in.

   .. code-block:: bash

      ? How would you like to authenticate BentoML CLI? [Use arrows to move]
      > Create a new API token with a web browser
        Paste an existing API token

3. Alternatively, you can log in by setting the ``api-token`` parameter if you already have an available token.

   .. code-block:: bash

      bentoml cloud login --api-token <your-api-token>

   .. note::

      The above command is displayed automatically after you create a token.

   Expected output:

   .. code-block:: bash

      Successfully logged in as user "user" in organization "mybentocloud".

4. To retrieve the current endpoint and API token locally, make sure you have installed ``jq``, and then run:

   .. code-block:: bash

      bentoml cloud current-context | jq '("endpoint:" + .endpoint + ", api_token:" + .api_token)'

After you log in, you should be able to manage BentoCloud resources. For more information on the CLI, see :doc:`/reference/bentocloud/bentocloud-cli`.

Access protected Deployments
----------------------------

You can use a token with **Protected Endpoint Access** to access a protected Bento Deployment. The following example provides different ways to interact with the :doc:`/get-started/hello-world` Summarization Service deployed with authorization enabled.

.. tab-set::

    .. tab-item:: CURL

        Include the token in the header of your HTTP request.

        .. code-block:: bash

            curl -s -X POST \
               'https://app-name.organization.cloud-apps.bentoml.com/summarize' \
               -H 'Authorization: Bearer $YOUR_TOKEN' \
               -H 'Content-Type: application/json' \
               -d '{
                  "text": "Your long text to summarize"
               }'

    .. tab-item:: Python client

        Set the ``token`` parameter in your :doc:`client </build-with-bentoml/clients>`.

        .. code-block:: python

            import bentoml

            client = bentoml.SyncHTTPClient("https://app-name.organization.cloud-apps.bentoml.com", token="******")
            response = client.summarize(text="Your long text to summarize")
            print(response)

    .. tab-item:: Browser

      To access a Protected Deployment from a web browser, you can add the token in the header using any browser extension that supports this feature, such as `Header Inject <https://chrome.google.com/webstore/detail/header-inject/cfmhknohjdjilpokjpdopankilegcglf>`_ in Google Chrome.

      1. Create a User token by following the steps in the :ref:`creating-an-api-token` section above. Make sure you select the desired Deployment that you want the token to access.
      2. Install Header Inject in Google Chrome and enable it.
      3. Select Header Inject, click **Add**, and specify **Header name** and **Header value**.

         .. image:: ../../_static/img/bentocloud/how-to/manage-access-tokens/header-inject.png
            :alt: Screenshot of the Header Inject browser extension interface showing how to add Authorization headers for accessing protected BentoML deployments

         - **Header name**: Enter ``Authorization``.
         - **Header value**: Enter ``Bearer $YOUR_TOKEN``.

      4. Click **Save**.
      5. Access the exposed URL of your Protected Deployment again and you should be able to access it.

Manage API tokens programmatically
-----------------------------------

You can manage API tokens programmatically using the BentoML Python SDK. This is useful for automation, CI/CD pipelines, and integrating token management into your workflows.

List API tokens
^^^^^^^^^^^^^^^

.. code-block:: python

    import bentoml

    # List all tokens
    tokens = bentoml.api_token.list()
    for token in tokens:
        print(f"{token.name}: {token.uid}")

    # Search for specific tokens
    tokens = bentoml.api_token.list(search="my-token")

Create an API token
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import bentoml
    from datetime import datetime, timedelta

    # Create a token with specific scopes
    token = bentoml.api_token.create(
        name="ci-cd-token",
        description="Token for CI/CD pipeline",
        scopes=["api", "read_cluster"],
        expired_at=datetime.now() + timedelta(days=30)
    )

    # Save the token value - it won't be shown again!
    print(f"Token value: {token.token}")

Available scopes:

- ``api``: General API access for managing BentoCloud resources
- ``read_organization``: Read organization data
- ``write_organization``: Write organization data
- ``read_cluster``: Read cluster data
- ``write_cluster``: Write cluster data

Get API token details
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import bentoml

    # Get token by UID
    token = bentoml.api_token.get("token_uid_here")
    if token:
        print(f"Name: {token.name}")
        print(f"Scopes: {token.scopes}")
        print(f"Expires: {token.expired_at}")
        print(f"Last used: {token.last_used_at}")

Delete an API token
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import bentoml

    # Delete a token by UID
    bentoml.api_token.delete("token_uid_here")

For the complete API reference, see :doc:`/reference/bentocloud/bentocloud-api`.

Manage API tokens using the CLI
--------------------------------

You can also manage API tokens using the BentoML CLI:

.. code-block:: bash

    # List all tokens
    bentoml api-token list

    # Create a new token
    bentoml api-token create my-token \
        --scope api --scope read_cluster \
        --expires 30d \
        --description "My CI/CD token"

    # Get token details
    bentoml api-token get <token_uid>

    # Delete a token
    bentoml api-token delete <token_uid>

For the complete CLI reference, see :doc:`/reference/bentocloud/bentocloud-cli`.

Use environment variables for API authentication
------------------------------------------------

When calling the BentoCloud API using Python, you can set the following environment variables for authentication:

- ``BENTO_CLOUD_API_KEY``: Your BentoCloud API token
- ``BENTO_CLOUD_API_ENDPOINT``: Your organization-specific BentoCloud endpoint

Example:

.. code-block:: bash

    export BENTO_CLOUD_API_KEY=cur7h***************
    export BENTO_CLOUD_API_ENDPOINT=https://organization_name.cloud.bentoml.com

.. note::

    When using environment variables, make sure to set both ``BENTO_CLOUD_API_KEY`` and ``BENTO_CLOUD_API_ENDPOINT`` as they are both required for authentication.

Monitoring API tokens
---------------------

BentoCloud provides a special type of API token called **Monitoring Tokens**, which are only available for BYOC (Bring Your Own Cloud) customers. These tokens are specifically designed for accessing Prometheus metrics in a federated manner.

.. note::

   Monitoring Tokens are disabled by default. If your organization needs access to them, please contact the BentoML team to have this feature enabled.

To use a Monitoring Token:

1. Create a Monitoring Token by following the steps in the :ref:`creating-an-api-token` section above, ensuring you select the `Monitoring Token` option.
2. Use the token to access Prometheus metrics through the federated endpoint:

   .. code-block:: bash

      curl -H "Authorization: Bearer $YOUR_TOKEN" \
           --get \
           --data-urlencode 'match[]={yatai_ai_bento_function!=""}' \
           https://prometheus.monitoring.$YOUR_CLUSTER.bentoml.ai/federate

This endpoint allows you to export metrics from your BentoML deployments to your own monitoring infrastructure.
