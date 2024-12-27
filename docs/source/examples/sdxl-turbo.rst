=========================
Stable Diffusion XL Turbo
=========================

Stable Diffusion XL Turbo (SDXL Turbo) is a distilled version of SDXL 1.0 and is capable of creating images in a single step, with improved real-time text-to-image output quality and sampling fidelity.

This document demonstrates how to create an image generation application with SDXL Turbo and BentoML.

All the source code in this tutorial is available in the `BentoDiffusion GitHub repository <https://github.com/bentoml/BentoDiffusion>`_.

Prerequisites
-------------

- Python 3.9+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/hello-world` first.
- To run this BentoML Service locally, you need a Nvidia GPU with at least 12G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

Clone the project repository and install all the dependencies.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoDiffusion.git
    cd BentoDiffusion/sdxl-turbo
    pip install -r requirements.txt

Create a BentoML Service
------------------------

Create a BentoML :doc:`Service </build-with-bentoml/services>` in a ``service.py`` file to define the serving logic of the model. You can use this example file in the cloned project:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from PIL.Image import Image

    MODEL_ID = "stabilityai/sdxl-turbo"

    sample_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    @bentoml.service(
        traffic={
            "timeout": 300,
            "external_queue": True,
            "concurrency": 1,
        },
        workers=1,
        resources={
            "gpu": 1,
            "gpu_type": "nvidia-l4",
        },
    )
    class SDXLTurbo:
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

In the Service code, the ``@bentoml.service`` decorator is used to define the ``SDXLTurbo`` class as a BentoML Service. It loads the pre-trained model (``MODEL_ID``) using the ``torch.float16`` data type. The model pipeline (``self.pipe``) is moved to a CUDA-enabled GPU device for efficient computation.

The ``txt2img`` method is an API endpoint that takes a text prompt, number of inference steps, and a guidance scale as inputs. It uses the model pipeline to generate an image based on the given prompt and parameters.

.. note::

   SDXL Turbo is capable of performing inference with just a single step. Therefore, setting ``num_inference_steps`` to ``1`` is typically sufficient for generating high-quality images. Additionally, you need to set ``guidance_scale`` to ``0.0`` to deactivate it as the model was trained without it. See `the official release notes <https://github.com/huggingface/diffusers/releases/tag/v0.24.0>`_ to learn more.

Run ``bentoml serve`` to start the BentoML server.

.. code-block:: bash

    $ bentoml serve service:SDXLTurbo

    2024-01-19T07:20:29+0000 [WARNING] [cli] Converting 'SDXLTurbo' to lowercase: 'sdxlturbo'.
    2024-01-19T07:20:29+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:SDXLTurbo" listening on http://localhost:3000 (Press CTRL+C to quit)

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

    .. tab-item:: Python client

        This client returns the image as a ``Path`` object. You can use it to access, read, or process the file. See :doc:`/build-with-bentoml/clients` for details.

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

        .. image:: ../../_static/img/examples/sdxl-turbo/service-ui.png

Expected output:

.. image:: ../../_static/img/examples/sdxl-turbo/output-image.png

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for your application. It is used for packaging your application into a Bento. Here is an example file in the project:

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:SDXLTurbo"
    labels:
      owner: bentoml-team
      project: gallery
    include:
    - "*.py"
    python:
      requirements_txt: "./requirements.txt"

:ref:`Log in to BentoCloud <scale-with-bentocloud/manage-api-tokens:Log in to BentoCloud using the BentoML CLI>` by running ``bentoml cloud login``, then run the following command to deploy the project.

.. code-block:: bash

    bentoml deploy .

Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

.. image:: ../../_static/img/examples/sdxl-turbo/sdxl-turbo-bentocloud.png

.. note::

   For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
