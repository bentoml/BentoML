========================
Run distributed Services
========================

BentoML provides a flexible framework for deploying machine learning models as Services. While a single Service often suffices for most use cases, it is useful to create multiple Services running in a distributed way in more complex scenarios.

This document provides guidance on creating and deploying a BentoML project with distributed Services.

Single and distributed Services
-------------------------------

Using a single BentoML Service in ``service.py`` is typically sufficient for most use cases. This approach is straightforward, easy to manage, and works well when you only need to deploy a single model and the API logic is simple.

In deployment, a BentoML Service runs as multiple processes in a container. If you define multiple Services, they run as processes in different containers. This distributed approach is useful when dealing with more complex scenarios, such as:

- **Pipelining CPU and GPU processing for better throughput**: Distributing tasks between CPU and GPU can enhance throughput. Certain preprocessing or postprocessing tasks might be more efficiently handled by the CPU, while the GPU focuses on model inference.
- **Optimizing resource utilization and scalability**: Distributed Services can run on different instances, allowing for independent scaling and efficient resource usage. This flexibility is important in handling varying loads and optimizing specific resource demands.
- **Asymmetrical GPU requirements**: Different models might have varied GPU requirements. Distributing these models across Services helps you allocate resources more efficiently and cost-effectively.
- **Handling complex workflows**: For applications involving intricate workflows, like sequential processing, parallel processing, or the composition of multiple models, you can create multiple Services to modularize these processes if necessary, improving maintainability and efficiency.

Interservice communication
--------------------------

Distributed Services support complex, modular architectures through interservice communication. Different Services can interact with each other using the ``bentoml.depends()`` function. This allows for direct method calls between Services as if they were local class functions. Key features of interservice communication:

- **Automatic service discovery & routing**: When Services are deployed, BentoML handles the discovery of Services, routes requests appropriately, and manages payload serialization and deserialization.
- **Arbitrary dependency chains**: Services can form dependency chains of any length, enabling intricate Service orchestration.
- **Diamond-shaped dependencies**: Support a diamond dependency pattern, where multiple Services depend on a single downstream Service, for maximizing Service reuse.

The following is an example of two distributed Services with different hardware requirements and one Service depends on another using ``bentoml.depends()``.

- ``SDXLControlNetService``: A high-resource demanding Service, requiring GPU support for image generation.
- ``ControlNet``: A Service designed to handle incoming requests and route them appropriately. It calls a method of the ``SDXLControlNetService`` Service.

This is the ``service.py`` file:

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
        resources={
            "gpu": "1",
            "gpu_type": "nvidia-l4",
        }
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
        traffic={"timeout": 600},
        workers=8,
    resources={"cpu": "1"}
    )
    class ControlNet:
        # Pass the dependent Service class as an argument
        controlnet_service = bentoml.depends(SDXLControlNetService)

        @bentoml.api
        async def generate(self, image: PIL_Image, params: Params) -> PIL_Image:
            arr = np.array(image)
            arr = cv2.Canny(arr, 100, 200)
            arr = arr[:, :, None]
            arr = np.concatenate([arr, arr, arr], axis=2)
            params_d = params.dict()
            prompt = params_d.pop("prompt")
            # Invoke a class level function of another Service
            res = await self.controlnet_service.generate(
                prompt,
                arr=arr,
                **params_d
            )
            return res[0][0]

To declare a dependency, you use the ``bentoml.depends()`` function by passing the dependent Service class as an argument. This creates a direct link between Services, facilitating easy method invocation. This example uses the following code to achieve this:

.. code-block:: python

    class ControlNet:
        controlnet_service = bentoml.depends(SDXLControlNetService)

Once a dependency is declared, invoking methods on the dependent Service is similar to calling a local method. In other words, Service ``A`` can call Service ``B`` as if Service ``A`` were invoking a class level function on Service ``B``. This abstracts away the complexities of network communication, serialization, and deserialization. In this example, the Service ``ControlNet`` invokes the ``generate`` function of ``SDXLControlNetService`` as below:

.. code-block:: python

    res = await self.controlnet_service.generate(prompt, arr=arr, **params_d)

Using ``bentoml.depends()`` is a recommended way for creating a BentoML project with distributed Services. It enhances modularity as you can develop reusable, loosely coupled Services that can be maintained and scaled independently.

Depend on an external deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BentoML also allows you to set an external deployment as a dependency for a Service. This means the Service can call a remote model and its exposed API endpoints. To specify an external deployment, use the ``bentoml.depends()`` function, either by providing the deployment name on BentoCloud or the URL if it's already running.

.. tab-set::

   .. tab-item:: Specify the Deployment name on BentoCloud

      You can also pass the ``cluster`` parameter to specify the cluster where your Deployment is running.

      .. code-block:: python

         import bentoml

         @bentoml.service
         class MyService:
             # `cluster` is optional if your Deployment is in a non-default cluster
             iris = bentoml.depends(deployment="iris-classifier-x6dewa", cluster="my_cluster_name")

             @bentoml.api
             def predict(self, input: np.ndarray) -> int:
                 # Call the predict function from the remote Deployment
                 return int(self.iris.predict(input)[0][0])

   .. tab-item:: Specify the URL

      If the external deployment is already running and its API is exposed via a public URL, you can reference it by specifying the ``url`` parameter. Note that ``url`` and ``deployment``/``cluster`` are mutually exclusive.

      .. code-block:: python

         import bentoml

         @bentoml.service
         class MyService:
             # Call the model deployed on BentoCloud by specifying its URL
             iris = bentoml.depends(url="https://<iris.example-url.bentoml.ai>")

             # Call the model served elsewhere
             # iris = bentoml.depends(url="http://192.168.1.1:3000")

             @bentoml.api
             def predict(self, input: np.ndarray) -> int:
                 # Make a request to the external service hosted at the specified URL
                 return int(self.iris.predict(input)[0][0])

.. tip::

   We recommend you specify the class of the external Service when using ``bentoml.depends()``. This makes it easier to validate the types and methods available on the remote Service.

   .. code-block:: python

      import bentoml

      @bentoml.service
      class MyService:
          # Specify the external Service class for type-safe integration
          iris = bentoml.depends(IrisClassifier, deployment="iris-classifier-x6dewa", cluster="my_cluster")

``bentofile.yaml``
------------------

For projects with multiple Services, you should reference the primary Service handling user requests for the ``service`` field in ``bentofile.yaml``. For example:

.. code-block:: yaml

    service: "service:ControlNet" #ControlNet is the one that receives users' requests
    labels:
      owner: bentoml-team
      project: gallery
    ...

You can then :doc:`containerize it as an OCI-compliant image </guides/containerization>` or deploy it to `BentoCloud <https://www.bentoml.com/>`_.

Deploy distributed Services
---------------------------

Deploying a project with distributed Services to BentoCloud is similar to deploying a single Service, with nuances in setting custom configurations.

To set custom configurations for each, we recommend you use a separate configuration file and reference it in the BentoML CLI command or Python API for deployment.

The following is an example file that defines some custom configurations for the above two Services. You set configurations of each Service in the ``services`` field. Refer to :doc:`/scale-with-bentocloud/deployment/configure-deployments` to see the available configuration fields.

.. code-block:: yaml

    # config-file.yaml
    name: "deployment-name"
    description: "This project creates an image generation application based on users' requirements."
    envs: # Optional. If you specify environment variables here, they will be applied to all Services
      - name: "GLOBAL_ENV_VAR_NAME"
        value: "env_var_value"
    services: # Add the configs of each Service under this field
      SDXLControlNetService: # Service one
        instance_type: "gpu.l4.1"
        scaling:
          max_replicas: 2
          min_replicas: 1
        envs: # Environment variables specific to Service one
          - name: "ENV_VAR_NAME"
            value: "env_var_value"
        deployment_strategy: "RollingUpdate"
        config_overrides:
          traffic:
            # float in seconds
            timeout: 700
            max_concurrency: 20
            external_queue: true
          resources:
            cpu: "400m"
            memory: "1Gi"
          workers:
            - gpu: 1
      ControlNet: # Service two
        instance_type: "cpu.1"
        scaling:
          max_replicas: 5
          min_replicas: 1

To deploy this Service to :doc:`BentoCloud </bentocloud/get-started>`, you can choose either the BentoML CLI or Python API:

.. tab-set::

    .. tab-item:: BentoML CLI

        .. code-block:: bash

            bentoml deploy . -f config-file.yaml

    .. tab-item:: Python API

        .. code-block:: python

            import bentoml
            bentoml.deployment.create(bento = "./path_to_your_project", config_file="config-file.yaml")
