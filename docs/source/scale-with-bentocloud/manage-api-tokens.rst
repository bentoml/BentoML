=================
Manage API tokens
=================

In BentoCloud, API tokens serve as a key method of authorization. You may use tokens to:

- Log in to BentoCloud
- Manage BentoCloud resources
- Access protected Deployments, which have :ref:`scale-with-bentocloud/deployment/configure-deployments:Authorization` enabled

This tutorial explains how to create and use API tokens in BentoCloud.

.. _creating-an-api-token:

Create an API token
===================

1. Click your profile image in the top-right corner of any BentoCloud page, then select **API Tokens**.
2. Click **Create**.
3. In the dialog that appears, specify the following fields. Note that you must select at least one of the token types.

   .. image:: ../../_static/img/bentocloud/how-to/manage-access-tokens/token-creation-dialog.png

   - **Name**: The name of the API token.
   - **Description**: A description of the token, detailing its usage.
   - **Developer Operations Access**: Grant permissions to access BentoCloud and manage resources on it.
   - **Protected Endpoint Access**: Grant permissions to access Bento Deployments with Protected endpoints. If you select this token type, you need to choose the Deployment that you want the token to access. If you want to use the token to access all the Protected Deployments, select **All Deployments**.
   - **Expired At**: Set an expiration date for the token. You won't be able to use the token after it expires.

4. Click **Submit**.
5. Record the token. This is the only opportunity to record it.
6. All available tokens appear on the **My API Tokens** page. Click **Delete** if you no longer need a token.

Log in to BentoCloud using the BentoML CLI
==========================================

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

After you log in, you should be able to manage BentoCloud resources. For more information on the CLI, see :doc:`Reference - CLI </reference/cli>`.

Access protected Deployments
============================

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

         - **Header name**: Enter ``Authorization``.
         - **Header value**: Enter ``Bearer $YOUR_TOKEN``.

      4. Click **Save**.
      5. Access the exposed URL of your Protected Deployment again and you should be able to access it.
