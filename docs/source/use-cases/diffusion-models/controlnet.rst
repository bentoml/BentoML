==========
ControlNet
==========

ControlNet is a neural network architecture designed to enhance the precision and control in generating images using text and image prompts. It allows you to influence image composition, adjust specific elements, and ensure spatial consistency. ControlNet can be used for various creative and precise image generation tasks, such as defining specific poses for human figures and replicating the composition or layout from one image in a new image.

This document demonstrates how to use ControlNet and Stable Diffusion XL to create an image generation application for specific user requirements.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

.. code-block:: bash

    pip install "bentoml>=1.2.0a0" torch transformers diffusers accelerate xformers opencv-python Pillow

Define the model serving logic
------------------------------

Create BentoML :doc:`/guides/services` in a ``service.py`` file to specify the serving logic of this BentoML project, which uses the following models:

- `diffusers/controlnet-canny-sdxl-1.0 <https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0>`_: Offers enhanced control in the image generation process. It allows for precise modifications based on text and image inputs, making sure the generated images are more aligned with specific user requirements (for example, replicating certain compositions).
- `madebyollin/sdxl-vae-fp16-fix <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix>`_: This Variational Autoencoder (VAE) is responsible for encoding and decoding images within the pipeline.
- `stabilityai/stable-diffusion-xl-base-1.0 <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`_: Takes text prompts and image inputs, processes them through the above two integrated models, and generates images that reflect the given prompts.

Here is an example of ``service.py``:

.. code-block:: python

    from __future__ import annotations

    import typing as t

    import cv2
    import numpy as np
    import PIL
    from PIL.Image import Image as PIL_Image

    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from pydantic import BaseModel

    import bentoml

    CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
    VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
    BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


    @bentoml.service(
        traffic={"timeout": 600},
        workers=1,
        resources={"gpu": "1", "memory": "16Gi"}
    )
    class SDXLControlNetService:

        def __init__(self) -> None:

            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.float16
            else:
                self.device = "cpu"
                self.dtype = torch.float32

            self.controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL_ID,
                torch_dtype=self.dtype,
            )

            self.vae = AutoencoderKL.from_pretrained(
                VAE_MODEL_ID,
                torch_dtype=self.dtype,
            )

            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                BASE_MODEL_ID,
                controlnet=self.controlnet,
                vae=self.vae,
                torch_dtype=self.dtype
            ).to(self.device)


        @bentoml.api
        async def generate(
                self,
                prompt: str,
                arr: np.ndarray[t.Any, np.uint8],
                **kwargs,
        ):
            image = PIL.Image.fromarray(arr)
            return self.pipe(prompt, image=image, **kwargs).to_tuple()


    class Params(BaseModel):
        prompt: str
        negative_prompt: t.Optional[str]
        controlnet_conditioning_scale: float = 0.5
        num_inference_steps: int = 25


    @bentoml.service(
        name="sdxl-controlnet-service",
        traffic={"timeout": 600},
        workers=8,
        resources={"cpu": "1"}
    )
    class APIService:
        controlnet_service: SDXLControlNetService = bentoml.depends(SDXLControlNetService)

        @bentoml.api
        async def generate(self, image: PIL_Image, params: Params) -> PIL_Image:
            arr = np.array(image)
            arr = cv2.Canny(arr, 100, 200)
            arr = arr[:, :, None]
            arr = np.concatenate([arr, arr, arr], axis=2)
            params_d = params.dict()
            prompt = params_d.pop("prompt")
            res = await self.controlnet_service.generate(
                prompt,
                arr=arr,
                **params_d
            )
            return res[0][0]

This file defines the following classes:

* ``SDXLControlNetService``: A BentoML Service with custom configurations in timeout, worker count, and resources.

  - It loads the three pre-trained models and configures them to use GPU if available. The main pipeline (``StableDiffusionXLControlNetPipeline``) integrates these models.
  - It defines an API endpoint ``generate`` to process a text prompt and an image array. The processed image is converted to a tuple and returned.

* ``Params``: This is a ``pydantic`` model defining the structure for input parameters.
* ``APIService``: A BentoML Service with custom configurations in timeout, worker count, and resources. ``APIService`` doesn't create images itself. Instead, it preprocesses the image and forwards it along with the text prompt to the ``SDXLControlNetService`` Service. The ``generate`` method in ``APIService`` then returns the final generated image.

Run ``bentoml serve`` in your project directory to start the BentoML server.

.. code-block:: bash

    $ bentoml serve service:APIService

    2024-01-09T04:33:24+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:APIService" can be accessed at http://localhost:3000/metrics.
    2024-01-09T04:33:24+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:APIService" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/generate' \
                -H 'accept: image/*' \
                -H 'Content-Type: multipart/form-data' \
                -F 'image=@example-image.png;type=image/png' \
                -F 'params={
                "prompt": "A young man walking in a park, wearing jeans.",
                "negative_prompt": "ugly, disfigured, ill-structured, low resolution",
                "controlnet_conditioning_scale": 0.5,
                "num_inference_steps": 25
                }'

    .. tab-item:: Python client

        .. code-block:: python

            import bentoml
            from pathlib import Path

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                result = client.generate(
                    image=Path("example-image.png"),
                    params={
                            "prompt": "A young man walking in a park, wearing jeans.",
                            "negative_prompt": "ugly, disfigured, ill-structure, low resolution",
                            "controlnet_conditioning_scale": 0.5,
                            "num_inference_steps": 25
                    },
                )

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, specify the image and parameters, and click **Execute**.

        .. image:: ../../_static/img/use-cases/diffusion-models/controlnet/service-ui.png

This is the example image used in the request:

.. image:: ../../_static/img/use-cases/diffusion-models/controlnet/example-image.png

Expected output:

.. image:: ../../_static/img/use-cases/diffusion-models/controlnet/output-image.png

Deploy the project to BentoCloud
--------------------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability.

First, specify a configuration YAML file (``bentofile.yaml``) as below to define the build options for your application. It is used for packaging your application into a Bento.

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:APIService"
    labels:
      owner: bentoml-team
      project: gallery
    include:
    - "*.py"
    python:
      requirements_txt: "./requirements.txt" # Put the installed dependencies into a separate requirements.txt file
    docker:
        distro: debian
        system_packages:
          - ffmpeg

Make sure you :doc:`have logged in to BentoCloud </bentocloud/how-tos/manage-access-token>`, then run the following command in your project directory to deploy the application to BentoCloud. Under the hood, this commands automatically builds a Bento, push the Bento, and deploy it on BentoCloud.

.. code-block:: bash

    bentoml deploy .

Once the application is up and running on BentoCloud, you can access it via the exposed URL.
