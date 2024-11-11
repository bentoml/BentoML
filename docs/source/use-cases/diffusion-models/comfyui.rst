=======
ComfyUI
=======

`ComfyUI <https://github.com/comfyanonymous/ComfyUI>`_ is a powerful tool for designing advanced diffusion pipelines. However, once the pipelines are built, deploying and serving them as API endpoints can be challenging and not very straightforward.

Recognizing the complexity of ComfyUI, BentoML provides a non-intrusive solution to serve existing ComfyUI pipelines as APIs without requiring any pipeline rewrites. It also offers the flexibility to customize the API endpoint's schema and logic.

Prerequisites
-------------

Install BentoML and its ComfyUI extension. The extension augments BentoML's command line interface with additional commands to save the ComfyUI workspace as a BentoML model and serve it as a Bento API endpoint. Installing the ``bentoml-comfyui`` package enables the ``bentoml comfyui`` sub-command.

.. code-block:: bash

    pip install bentoml bentoml-comfyui

Ensure the ComfyUI workflow is functional in your environment. Include only the models and custom nodes required for running the workflow. Including unused models and custom nodes will increase the size of the model unnecessarily and slows down deployment and cold-start. It is also recommended to save and reload the workflow API JSON file to ensure it is working as expected.

.. note::

    It is recommended to develop in a Ubuntu/Debian environment for dependency resolution and compatibility.

Annotate the workflow with input and output
-------------------------------------------

One challenge of serving ComfyUI pipelines as APIs is the lack of a standard schema for the input and output of the workflow. To address this, BentoML published a suite of lightweight ComfyUI nodes to indicate the input and ouput. These nodes are non-intrusive and do not affect the workflow's functionality. Search for "ComfyUI-IDL" in the ComfyUI Manager to install.

1. Identify the widget you'd like to expose as the input by converting it as an input field. Right-click a node and select "Convert Widget to Input" then select the widget name.

    .. image:: ../../_static/img/use-cases/diffusion-models/comfyui/convert_widget.png

2. Add an input node of the corresponding type by right-clicking the background and selecting "Add Node", "ComfyUI-IDL", "input", then the node type. Connect the input to the widget converted in step 1.

    .. image:: ../../_static/img/use-cases/diffusion-models/comfyui/add_input_node.png

3. Add an output node by following the same steps as step 2, but selecting "output" instead of "input". Connect the output to the output node.

Save the workflow API
---------------------

Rerun the workflow and ensure it is working as expected after the addition of the input and output nodes. Save the workflow API as a JSON file by clicking the "Save (API Format)" with develop mode enabled.

.. note::

    To enable developer mode, first click the gear icon in the top right corner of the menu then toggle on the "Enable dev mode options (API save, etc)" option. "Save (API Format)" button should now appear in the menu.

Save the workspace as a model
-----------------------------

A ComfyUI workflow has complex model and custom node dependencies. To ensure the workflow runs correctly in a different environment, BentoML saves the entire workspace as a model.

.. code-block:: bash

    bentoml comfyui pack --name my-comfyui-workspace [WORKSPACE_PATH]

Now we have the workspace saved as a BentoML model. You can confirm it by running ``bentoml models list``.

Build the workflow as a Bento
-----------------------------

In this step, we build the workspace and workflow as a Bento. We require the workflow API JSON file and the workspace model we saved in the previous steps. Underneath the hood, BentoML freezes the current Python virtual environment as dependency and genereates a service module with the serving logic for the workflow.

.. code-block:: bash

    bentoml comfyui build workflow_api.json --model my-comfyui-workspace --name my-comfyui-bento

Many ComfyUI workflows requires system packages such as ``git`` and ``ffmpeg``. Use ``--system-packages`` to specify the required system packages.

.. code-block:: bash

    bentoml comfyui build workflow_api.json --model my-comfyui-workspace --name my-comfyui-bento --system-packages git --system-pacakges ffmpeg

Verify the behavior of the Bento by running it locally.

.. code-block:: bash

    bentoml serve my-comfyui-bento:latest

Deploy the workflow a Bento
---------------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits. Follow the instructions to `log in <bentocloud/how-tos/manage-access-token:Log in to BentoCloud using the BentoML CLI>` to BentoCloud.

   .. code-block:: bash

        bentoml cloud login

Deploy the Bento created in the previous step to BentoCloud by running the following command.

.. code-block:: bash

    bentoml deploy consistent-character:fhe7tcvadkqhar7j
