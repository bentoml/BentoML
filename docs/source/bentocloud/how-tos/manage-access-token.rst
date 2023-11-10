====================
Manage access tokens
====================

In BentoCloud, API tokens serve as a key method of authorization for two distinct scopes - **BentoCloud resources** and **Bento Deployments**.
They correspond to two different types of tokens - **User tokens** and **Developer tokens**.

User tokens are granted permissions to access deployed Bento applications. You can control access to Bento Deployments with the following endpoint access types.

- **Protected**: The Deployment is accessible to anyone on the internet, provided that they have a valid token.
- **Public**: The Deployment is accessible to anyone on the internet.

.. note::

   You can specify the endpoint access type when creating and updating a Deployment.

Developer tokens are granted permissions to manage BentoCloud resources. For example, you can perform the following tasks with a Developer token:

- Manage BentoCloud cluster configurations.
- Handle identity and access management (IAM) policies.
- Manage models, Bentos, and Deployments.

This tutorial explains how to create and use API tokens in BentoCloud.

.. _creating-an-api-token:

Create an API token
===================

To create an API token, perform the following steps:

1. Navigate to the `API Tokens <http://cloud.bentoml.com/api_tokens>`_ page in the BentoCloud Console.
2. Click **Create**.
3. In the dialog that appears, specify the following fields. Note that you must select at least one of the token types.

   .. image:: ../../_static/img/bentocloud/how-to/manage-access-tokens/token-creation-dialog.png

   - **Name**: The name of the API token.
   - **Description**: A description of the token, detailing its usage.
   - **Developer Operations Access (Developer token)**: Grant permissions to access BentoCloud and manage resources on it.
   - **Protected Endpoint Access (User token)**: Grant permissions to access Bento Deployments with Protected endpoints. If you select this token type, you need to choose the Deployment that you want the token to access. If you want to use the token to access all the Protected Deployments, select **All Deployments**.
   - **Expired At**: Set an expiration date for the token. You won't be able to use the token after it expires.

4. Click **Submit**.
5. Record the token. This is the only opportunity to record it.
6. All available tokens appear on the **API Tokens** page. Click **Delete** if you no longer needs a token.

Use the Developer token
=======================

Interact with BentoCloud programmatically via the BentoML Command Line
Interface (CLI). Log in using the following command.

.. code-block:: bash

   bentoml cloud login --api-token <your-api-token> --endpoint <your-bentocloud-endpoint>

.. note::

   You should see the above command after you create a token.

Expected output:

.. code-block:: bash

   Successfully logged in as user "user" in organization "mybentocloud".

To retrieve the current endpoint and API token locally, make sure you have installed ``jq``, and then run:

.. code-block:: bash

   bentoml cloud current-context | jq '("endpoint:" + .endpoint + ", api_token:" + .api_token)'

After you log in, you should be able to manage BentoCloud resources. For more information on the CLI, see :doc:`Reference - CLI </reference/cli>`.

Use the User token
==================

You can use User tokens to access Protected Bento Deployments.

For HTTP-based servers, include the token in the header of your HTTP request.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl "http://app-name.organization.cloud-apps.bentoml.com" \
               -H "Content-Type: application/json" \
               -H "Authorization: Bearer $YOUR_TOKEN" \
               --data '{"prompt": "What state is Los Angeles in?", "llm_config": {"max_new_tokens": 129}}'

    .. tab-item:: Browser

      To access a Protected Deployment from a web browser, you can add the token in the header using any browser extension that supports this feature, such as `Header Inject <https://chrome.google.com/webstore/detail/header-inject/cfmhknohjdjilpokjpdopankilegcglf>`_ in Google Chrome.

      1. Create a User token by following the steps in the :ref:`creating-an-api-token` section above. Make sure you select the desired Deployment that you want the token to access.
      2. Install Header Inject in Google Chrome and enable it.
      3. Select Header Inject, click **Add**, and specify **Header name** and **Header value**.

         .. image:: ../../_static/img/bentocloud/how-to/manage-access-tokens/header-inject.png

         - **Header name**: Enter ``Authorization``.
         - **Header value**: Enter ``Bearer $YOUR_TOKEN``.

      4. Click **Save**.
      5. Access the exposed URL of your Protected Deployment again and you should be able to access it.

For gRPC servers, include it in the metadata of your gRPC call:

.. code-block:: python

   import grpc

   creds = grpc.ssl_channel_credentials()
   auth_creds = grpc.access_token_call_credentials('<your-api-token>')
   channel = grpc.secure_channel('<your-deployed-api-endpoint>', creds)
   stub = <YourGRPCServiceStub>(channel)
