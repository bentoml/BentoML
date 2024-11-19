=========================
Call Deployment endpoints
=========================

This document explains how to interact with a Deployment by calling its endpoint URL.

Obtain the endpoint URL
-----------------------

Choose one of the following ways to obtain the endpoint URL.

.. tab-set::

    .. tab-item:: BentoML CLI

        Install `jq <https://github.com/jqlang/jq>`_, then run the following command.

        .. code-block:: bash

            bentoml deployment get <your_deployment_name> -o json | jq ."endpoint_urls"

    .. tab-item:: Python API

        .. code-block:: python

            import bentoml

            deployment_info = bentoml.deployment.get('your_deployment_name')
            print(deployment_info.get_endpoint_urls())

    .. tab-item:: BentoCloud console

        1. Navigate to the **Deployments** page.
        2. Click the desired Deployment.
        3. On the details page, you can find the endpoint URL under the Deployment's name.

           .. image:: ../../_static/img/bentocloud/how-to/call-deployment-endpoints/deployment-endpoint-url.png

Interact with the Deployment
----------------------------

Choose one of the following ways to access your Deployment with the endpoint URL. The example below shows how to interact with the :ref:`Summarization Deployment <bentocloud/get-started:deploy your first model>`. You can find the corresponding Python code and CURL command for a specific Deployment on the **Playground** tab of its details page.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'https://<deployment_endpoint_url>/summarize' \
                -H 'accept: text/plain' \
                -H 'Content-Type: application/json' \
                -d '{
                "text": "Your long text to summarize"
                }'

    .. tab-item:: Python client

        Include the endpoint URL in your client as below. For more information, see :doc:`/build-with-bentoml/clients`.

        .. code-block:: python

            import bentoml

            client = bentoml.SyncHTTPClient(url='<deployment_endpoint_url>')
            result: str = client.summarize(text="Your long text to summarize")
            print(result)

        You can retrieve the information of a client by using ``get_client`` or ``get_async_client`` (set the ``token`` parameter if you enable :ref:`scale-with-bentocloud/deployment/configure-deployments:authorization`), then call its available endpoint to make HTTP requests to your Deployment.

        .. code-block:: python

            import bentoml

            dep = bentoml.deployment.get(name="deploy-1")
            # Get synchronous HTTP client for Deployment:
            client = dep.get_client()
            # Get asynchronous HTTP client for Deployment:
            async_client = dep.get_async_client()
            # Call the client's endpoint to interact
            result = client.summarize(text="Your long text to summarize")

    .. tab-item:: Swagger UI

        Access the Deployment endpoint URL directly. The Swagger UI dynamically generates documentation and a user interface based on OpenAPI Specifications.

        .. image:: ../../_static/img/bentocloud/how-to/call-deployment-endpoints/swagger-ui.png

Interact with protected endpoints
---------------------------------

If you enable :ref:`scale-with-bentocloud/deployment/configure-deployments:authorization` for a Deployment when creating it, its endpoint URL will be protected. You need to create :ref:`an API token with Protected Endpoint Access <scale-with-bentocloud/manage-api-tokens:create an api token>` and then :ref:`use this token to access it <scale-with-bentocloud/manage-api-tokens:access protected deployments>`.
