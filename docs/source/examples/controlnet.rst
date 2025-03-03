==========
ControlNet
==========

ControlNet is a neural network architecture designed to enhance the precision and control in generating images using text and image prompts. It allows you to influence image composition, adjust specific elements, and ensure spatial consistency. ControlNet can be used for various creative and precise image generation tasks, such as defining specific poses for human figures and replicating the composition or layout from one image in a new image.

This document demonstrates how to use ControlNet and Stable Diffusion XL to create an image generation application for specific user requirements.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoDiffusion" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
        </div>
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-left: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/bentocloud-logo.png" alt="BentoCloud" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="#bentocloud" style="margin-left: 5px; vertical-align: middle;">Deploy to BentoCloud</a>
        </div>
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-left: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/bentoml-icon.png" alt="BentoML" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="#localserving" style="margin-left: 5px; vertical-align: middle;">Serve with BentoML</a>
        </div>
    </div>

You can invoke the ControlNet inference API with parameters specifying your desired image characteristics. For example, send the following query to generate a new scene replicating the pose from the provided reference image:

.. code-block:: bash

     {
        "prompt": "A young man walking in a park, wearing jeans.",
        "negative_prompt": "ugly, disfigured, ill-structured, low resolution",
        "controlnet_conditioning_scale": 0.5,
        "num_inference_steps": 25,
        "image": "example-image.png",
     }

Input reference image:

.. image:: ../../_static/img/examples/controlnet/example-image.png
   :align: center
   :width: 400px

This is the generated output image, replicating the pose in a new environment:

.. image:: ../../_static/img/examples/controlnet/output-image.png
   :align: center
   :width: 400px

This example is ready for quick deployment and scaling on BentoCloud. With a single command, you get a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/controlnet/controlnet-bentocloud.png

Code explanations
-----------------

You can find `the source code in GitHub <https://github.com/bentoml/BentoDiffusion/tree/main/controlnet>`_. Below is a breakdown of the key code implementations within this project.

1. Set the model IDs used by the ControlNet and SDXL pipeline. You can switch to any other diffusion model as needed.

   - `diffusers/controlnet-canny-sdxl-1.0 <https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0>`_: Offers enhanced control in the image generation process. It allows for precise modifications based on text and image inputs, making sure the generated images are more aligned with specific user requirements (for example, replicating certain compositions).
   - `madebyollin/sdxl-vae-fp16-fix <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix>`_: This Variational Autoencoder (VAE) is responsible for encoding and decoding images within the pipeline.
   - `stabilityai/stable-diffusion-xl-base-1.0 <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`_: Takes text prompts and image inputs, processes them through the above two integrated models, and generates images that reflect the given prompts.

   .. code-block:: python
      :caption: `service.py`

      CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
      VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
      BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

2. Use the ``@bentoml.service`` decorator to define a BentoML Service, where you can customize how the model will be served. The decorator lets you set :doc:`configurations </reference/bentoml/configurations>` like timeout and GPU resources to use on BentoCloud. Note that these models require at least an NVIDIA L4 GPU for optimal performance.

   .. code-block:: python
      :caption: `service.py`

      @bentoml.service(
            traffic={"timeout": 600},
            resources={
                "gpu": 1,
                "gpu_type": "nvidia-l4",
            }
      )
      class ControlNet:
          controlnet_path = bentoml.models.HuggingFaceModel(CONTROLNET_MODEL_ID)
          vae_path = bentoml.models.HuggingFaceModel(VAE_MODEL_ID)
          base_path = bentoml.models.HuggingFaceModel(BASE_MODEL_ID)
          ...

   Within the class, :ref:`load the model from Hugging Face <load-models>` and define it as a class variable. The ``HuggingFaceModel`` method provides an efficient mechanism for loading AI models to accelerate model deployment on BentoCloud, reducing image build time and cold start time.

3. The ``@bentoml.service`` decorator also allows you to :doc:`define the runtime environment </build-with-bentoml/runtime-environment>` for a Bento, the unified distribution format in BentoML. A Bento is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

   Here is an example:

   .. code-block:: python
      :caption: `service.py`

      my_image = bentoml.images.PythonImage(python_version="3.11", distro="debian") \
                  .system_packages("ffmpeg") \
                  .requirements_file("requirements.txt")

      @bentoml.service(
          image=my_image, # Apply the specifications
          ...
      )
      class ControlNet:
          ...

4. Use the ``@bentoml.api`` decorator to define an asynchronous API endpoint ``generate``. It takes an image and a set of parameters as input, and returns the generated image by calling the pipeline with the processed image and text prompts.

   .. code-block:: python
      :caption: `service.py`

      class ControlNet:
          ...

          def __init__(self) -> None:

              import torch
              from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
              # Logic to initialize models here
              ...

          @bentoml.api
          def generate(
                  self,
                  image: PIL_Image,
                  prompt: str,
                  negative_prompt: t.Optional[str] = None,
                  controlnet_conditioning_scale: t.Optional[float] = 1.0,
                  num_inference_steps: t.Optional[int] = 50,
                  guidance_scale: t.Optional[float] = 5.0,
          ) -> PIL_Image:
              ...
              return self.pipe(
                  prompt,
                  image=image,
                  negative_prompt=negative_prompt,
                  controlnet_conditioning_scale=controlnet_conditioning_scale,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
              ).to_tuple()[0][0]

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoDiffusion/tree/main/controlnet>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image, and deploy it anywhere.

.. _BentoCloud:

BentoCloud
^^^^^^^^^^

.. raw:: html

    <a id="bentocloud"></a>

BentoCloud provides fast and scalable infrastructure for building and scaling AI applications with BentoML in the cloud.

1. Install BentoML and :doc:`log in to BentoCloud </scale-with-bentocloud/manage-api-tokens>` through the BentoML CLI. If you don't have a BentoCloud account, `sign up here for free <https://www.bentoml.com/>`_.

   .. code-block:: bash

      pip install bentoml
      bentoml cloud login

2. Clone the `BentoDiffusion repository <https://github.com/bentoml/BentoDiffusion>`_ and deploy the project.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoDiffusion.git
      cd BentoDiffusion/controlnet
      bentoml deploy

3. Once it is up and running on BentoCloud, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/controlnet/controlnet-bentocloud.png

    .. tab-item:: Python client

       Create a :doc:`BentoML client </build-with-bentoml/clients>` to call the endpoint. Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: python

          import bentoml
          from pathlib import Path

          # Define the path to save the generated image
          output_path = Path("generated_image.png")

          with bentoml.SyncHTTPClient("https://controlnet-new-testt-e3c1c7db.mt-guc1.bentoml.ai") as client:
            result = client.generate(
                controlnet_conditioning_scale=0.5,
                guidance_scale=5,
                image=Path("./example-image.png"),
                negative_prompt="ugly, disfigured, ill-structure, low resolution",
                num_inference_steps=25,
                prompt="A young man walking in a park, wearing jeans.",
          )

          # The result should be a PIL.Image object
          result.save(output_path)

          print(f"Image saved at {output_path}")

    .. tab-item:: CURL

       Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: bash

          curl -s -X POST \
                'https://controlnet-new-testt-e3c1c7db.mt-guc1.bentoml.ai/generate' \
                -F controlnet_conditioning_scale='0.5' \
                -F guidance_scale='5' \
                -F negative_prompt='"ugly, disfigured, ill-structure, low resolution"' \
                -F num_inference_steps='25' \
                -F prompt='"A young man walking in a park, wearing jeans."' \
                -F 'image=@example-image.png' \
                -o output.jpg

4. To make sure the Deployment automatically scales within a certain replica range, add the scaling flags:

   .. code-block:: bash

      bentoml deploy --scaling-min 0 --scaling-max 3 # Set your desired count

   If it's already deployed, update its allowed replicas as follows:

   .. code-block:: bash

      bentoml deployment update <deployment-name> --scaling-min 0 --scaling-max 3 # Set your desired count

   For more information, see :doc:`how to configure concurrency and autoscaling </scale-with-bentocloud/scaling/autoscaling>`.

.. _LocalServing:

Local serving
^^^^^^^^^^^^^

.. raw:: html

    <a id="localserving"></a>

BentoML allows you to run and test your code locally, so that you can quickly validate your code with local compute resources.

1. Clone the repository and choose your desired project.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoDiffusion.git
      cd BentoDiffusion/controlnet

      # Recommend Python 3.11
      pip install -r requirements.txt

2. Serve it locally.

   .. code-block:: bash

      bentoml serve

   .. note::

      To run this project locally, you need an Nvidia GPU with at least 12G VRAM.

3. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
