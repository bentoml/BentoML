=========================
Stable Diffusion XL Turbo
=========================

Stable Diffusion XL Turbo (SDXL Turbo) is a distilled version of SDXL 1.0 and is capable of creating images in a single step, with improved real-time text-to-image output quality and sampling fidelity.

This document demonstrates how to create an image generation application with SDXL Turbo and BentoML.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- To run this BentoML Service locally, you need a Nvidia GPU with at least 12G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

.. code-block:: bash

    pip install "bentoml>=1.2.0a0" torch transformers diffusers accelerate xformers Pillow

Create a BentoML Service
------------------------

Create a BentoML :doc:`Service </guides/services>` in a ``service.py`` file to define the serving logic of the model.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from PIL.Image import Image

    MODEL_ID = "stabilityai/sdxl-turbo"

    sample_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    @bentoml.service(
        traffic={"timeout": 300},
        workers=1,
        resources={"gpu": 1, "memory": "12Gi"},
    )
    class SDXLTurboService:
        def __init__(self) -> None:
            from diffusers import AutoPipelineForText2Image
            import torch

            self.pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            self.pipe.to(device="cuda")

        @bentoml.api
        def txt2img(
                self,
                prompt: str = sample_prompt,
                num_inference_steps: int = 1,
                guidance_scale: float = 0.0,
        ) -> Image:
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            return image

In the Service code, the ``@bentoml.service`` decorator is used to define the ``SDXLTurboService`` class as a BentoML Service. It loads the pre-trained model (``MODEL_ID``) using the ``torch.float16`` data type. The model pipeline (``self.pipe``) is moved to a CUDA-enabled GPU device for efficient computation.

The ``txt2img`` method is an API endpoint that takes a text prompt, number of inference steps, and a guidance scale as inputs. It uses the model pipeline to generate an image based on the given prompt and parameters.

.. note::

   SDXL Turbo is capable of performing inference with just a single step. Therefore, setting ``num_inference_steps`` to ``1`` is typically sufficient for generating high-quality images. Additionally, you need to set ``guidance_scale`` to ``0.0`` to deactivate it as the model was trained without it. See `the official release notes <https://github.com/huggingface/diffusers/releases/tag/v0.24.0>`_ to learn more.

Run ``bentoml serve`` to start the BentoML server.

.. code-block:: bash

    $ bentoml serve service:SDXLTurboService

    2024-01-19T07:20:29+0000 [WARNING] [cli] Converting 'SDXLTurboService' to lowercase: 'sdxlturboservice'.
    2024-01-19T07:20:29+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:SDXLTurboService" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/txt2img' \
                -H 'accept: image/*' \
                -H 'Content-Type: application/json' \
                --output output.png \
                -d '{
                "prompt": "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
                "num_inference_steps": 1,
                "guidance_scale": 0
            }'

    .. tab-item:: BentoML client

        This client returns the image as a ``Path`` object. You can use it to access, read, or process the file. See :doc:`/guides/clients` for details.

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                    result = client.txt2img(
                        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
                        num_inference_steps=1,
                        guidance_scale=0.0
                    )

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, specify the parameters, and click **Execute**.

        .. image:: ../../_static/img/use-cases/diffusion-models/sdxl-turbo/service-ui.png

Expected output:

.. image:: ../../_static/img/use-cases/diffusion-models/sdxl-turbo/output-image.png

Deploy the project to BentoCloud
--------------------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability.

First, specify a configuration YAML file (``bentofile.yaml``) as below to define the build options for your application. It is used for packaging your application into a Bento.

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:SDXLTurboService"
    labels:
      owner: bentoml-team
      project: gallery
    include:
    - "*.py"
    python:
      requirements_txt: "./requirements.txt" # Put the installed dependencies into a separate requirements.txt file

Make sure you :doc:`have logged in to BentoCloud </bentocloud/how-tos/manage-access-token>`, then run the following command in your project directory to deploy the application to BentoCloud. Under the hood, this commands automatically builds a Bento, push the Bento, and deploy it on BentoCloud.

.. code-block:: bash

    bentoml deploy .

Once the application is up and running on BentoCloud, you can access it via the exposed URL.
