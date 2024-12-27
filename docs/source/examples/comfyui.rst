=================================
ComfyUI: Deploy workflows as APIs
=================================

`ComfyUI <https://github.com/comfyanonymous/ComfyUI>`_ is a powerful tool for designing advanced diffusion workflows. It provides an extensive collection of resources, including shared workflows and custom nodes, to help creative workers and developers generate content without dealing with complex code. However, deploying and serving these workflows as scalable API endpoints can be `complex and non-intuitive <https://www.bentoml.com/blog/comfy-pack-serving-comfyui-workflows-as-apis>`_.

To address the deployment challenges, the BentoML team developed `comfy-pack <https://github.com/bentoml/comfy-pack>`_, a comprehensive toolkit that transforms ComfyUI workflows into production-grade APIs. Specifically, comfy-pack enables you to:

- Define standardized API schemas for workflow inputs and outputs
- Serve workflows as HTTP endpoints accessible via standard API clients
- Deploy workflows to BentoCloud with enterprise-grade features such as fast autoscaling and built-in observability
- Package the complete workspace as portable artifacts for consistent reproduction

Installation
------------

You can install comfy-pack using either `ComfyUI Manager <https://github.com/ltdrdata/ComfyUI-Manager>`_ or Git.

.. tab-set::

    .. tab-item:: ComfyUI Manager (Recommended)

        1. Open **ComfyUI Manager**.
        2. Search for ``comfy-pack`` and click **Install**.

           .. image:: ../../_static/img/examples/comfyui/install-comfy-pack-via-comfyui-manager.png
              :alt: Install comfy-pack via ComfyUI Manager

        3. Click **Restart** and refresh your ComfyUI interface to apply changes.

    .. tab-item:: Git

        Clone the repository into your ComfyUI custom nodes directory:

        .. code-block:: bash

           cd ComfyUI/custom_nodes
           git clone https://github.com/bentoml/comfy-pack.git

Specify input and output nodes
------------------------------

When serving ComfyUI workflows as APIs, one key challenge is establishing a standardized schema for workflow inputs and outputs. comfy-pack addresses this by providing dedicated interface nodes that integrate seamlessly with existing workflows without affecting their core functionality.

1. Right-click a node containing the widget you want to expose.
2. Select **Convert Widget to Input**, then choose the widget name.
3. To add a comfy-pack input node:

   a. Right-click anywhere on the blank space.

   b. Navigate to **Add Node > ComfyPack > input**, then select the desired input node type:

      - **Image Input**: Accepts image type input, similar to the official ``LoadImage`` node.
      - **String Input**: Accepts string type input (e.g., prompts).
      - **Int Input**: Accepts integer type input (e.g., dimensions, seeds).
      - **File Input**: Accepts file type input.
      - **Any Input**: Accepts combo type and other input (e.g., custom nodes).

4. Connect the comfy-pack input node to the widget you converted previously.

   .. image:: ../../_static/img/examples/comfyui/add-comfy-pack-input-node.gif
      :alt: Add comfy-pack input node

5. To add a comfy-pack output node:

   a. Right-click anywhere on the blank space.

   b. Navigate to **Add Node > ComfyPack > output**, then select the desired output node type:

      - **File Output**: Outputs a file path as a string and saves the file to the specified location.
      - **Image Output**: Outputs an image, similar to the official ``SaveImage`` node.

6. Connect the workflow's output to the comfy-pack output node.

   .. image:: ../../_static/img/examples/comfyui/add-comfy-pack-output-node.gif
      :alt: Add comfy-pack output node

7. Run the workflow to ensure it functions as expected.

Serve workflows as APIs
-----------------------

You can expose ComfyUI workflows as HTTP APIs that can be called from any client.

1. On the toolbar at the top of the screen, click **Serve**.
2. Set the desired port (default: ``3000``).
3. Click **Start** to launch the server. The API will be available at ``http://127.0.0.1:<port>``.
4. The server exposes a ``/generate`` endpoint. Use it to submit requests with parameters configured through comfy-pack nodes (e.g., ``prompt``, ``width``, ``height``, ``seed``). For example:

   .. tab-set::

      .. tab-item:: CURL

         .. code-block:: bash

            curl -X 'POST' \
                'http://127.0.0.1:3000/generate' \
                -H 'accept: application/octet-stream' \
                -H 'Content-Type: application/json' \
                --output output.png \
                -d '{
                "prompt": "rocks in a bottle",
                "width": 512,
                "height": 512,
                "seed": 1
            }'

      .. tab-item:: Python client

         comfy-pack uses BentoML as its serving framework, allowing you to use the :doc:`BentoML Python client </build-with-bentoml/clients>` for interaction:

         .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://127.0.0.1:3000") as client:
                result = client.generate(
                    prompt="rocks in a bottle",
                    width=512,
                    height=512,
                    seed=1
                )

   .. important::

      Parameter names in API calls must match your comfy-pack node names.

Deploy to BentoCloud
--------------------

You can deploy your ComfyUI workflow to BentoCloud for better management and scalability.

1. On the toolbar at the top of the screen, click **Deploy**.
2. In the dialog that appears, set a name and select required models and system packages.
3. Enter your BentoCloud access token. If you don't have a BentoCloud account, `sign up for free <https://www.bentoml.com/>`_ and :doc:`create a token </scale-with-bentocloud/manage-api-tokens>`.
4. Click **Push to Cloud** and wait for your Bento to be built.
5. Once it's ready, click **Deploy Now** to open the Bento details page on BentoCloud.
6. Deploy the Bento from the BentoCloud console.

Package and restore a workspace
-------------------------------

You can package a ComfyUI workspace into a portable artifact, ensuring it can be easily transferred and unpacked elsewhere with consistent behavior.

Create a package
^^^^^^^^^^^^^^^^

1. On the toolbar at the top of the screen, click **Package**.
2. Set a name for the package.
3. (Optional) Choose which models to include. Note that only model hashes are stored, not the actual files. This keeps package size minimal while ensuring version accuracy.
4. Click **Pack**. Your browser will automatically download a ``.cpack.zip`` file.

Restore a workspace
^^^^^^^^^^^^^^^^^^^

1. Install comfy-pack CLI:

   .. code-block:: bash

      pip install comfy-pack

2. Unpack the ``.cpack.zip`` file:

   .. code-block:: bash

      comfy-pack unpack <workflow_name>.cpack.zip

When unpacking, comfy-pack restores the original ComfyUI workspace by performing the following steps:

1. Prepares a Python virtual environment with the exact packages used in the workflow.
2. Clones the specific ComfyUI version and custom nodes, pinned to the exact versions required by the workflow.
3. Searches for and downloads models from common registries like Hugging Face and Civitai. It uses symbolic links for efficient model sharing (i.e., models are downloaded only once and reused across workflows) and verifies model integrity via hash checking.
