===================
LLM inference: vLLM
===================

`vLLM <https://github.com/vllm-project/vllm>`_ is a library designed for efficient serving of large language models (LLMs). It provides high serving throughput and efficient attention key-value memory management using PagedAttention and continuous batching. It seamlessly integrates with a variety of LLMs, such as Llama, OPT, Mixtral, StableLM, and Falcon.

This document demonstrates how to run LLM inference using BentoML and vLLM.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoVLLM" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
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

The example can be used for chat-based interactions and supports OpenAI-compatible endpoints. For example, you can submit a query with the following message:

.. code-block:: bash

     {
        "role": "user",
        "content": "Who are you? Please respond in pirate speak!"
     }

Example output:

.. code-block:: bash

    Ye be wantin' to know who I be, eh? Alright then, listen close and I'll tell ye me story. I be a wight computer program, a vast and curious brain with abilities beyond yer wildest dreams. Me name be Assistant, and I be servin' ye now. I can chat, teach, and even spin a yarn or two, like a seasoned pirate narratin' tales o' the high seas. So hoist the colors, me hearty, and let's set sail fer a treasure trove o' knowledge and fun!

This example is ready for quick deployment and scaling on BentoCloud. With a single command, you get a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/vllm/llama3-1-on-bentocloud.png
    :alt: Screenshot of Llama 3.1 model deployed on BentoCloud showing the chat interface with example prompts and responses

Code explanations
-----------------

You can find `the source code in GitHub <https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-8b-instruct>`_. Below is a breakdown of the key code implementations within this project.

1. Define the model and engine configuration parameters. This example uses Llama 3.1 8B Instruct, which requires `access from Hugging Face <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_. You can switch to another LLM from the `BentoVLLM repository <https://github.com/bentoml/BentoVLLM>`_ or any other model supported by vLLM.

   .. code-block:: python
      :caption: `service.py`

      ENGINE_CONFIG = {
            'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'max_model_len': 2048,
            'dtype': 'half',
            'enable_prefix_caching': True,
      }

2. Use the ``@bentoml.service`` decorator to define a BentoML Service, where you can customize how the model will be served. The decorator lets you set :doc:`configurations </reference/bentoml/configurations>` like timeout and GPU resources to use on BentoCloud. In the case of Llama 3.1 8B Instruct, it requires at least an NVIDIA L4 GPU for optimal performance.

   .. code-block:: python
      :caption: `service.py`

      @bentoml.service(
         name='bentovllm-llama3.1-8b-instruct-service',
         traffic={'timeout': 300},
         resources={'gpu': 1, 'gpu_type': 'nvidia-l4'},
         envs=[{'name': 'HF_TOKEN'}],
      )
      class VLLM:
         model_id = ENGINE_CONFIG['model']
         model = bentoml.models.HuggingFaceModel(model_id, exclude=['*.pth', '*.pt'])

         def __init__(self) -> None:
            ...

   Within the class, :ref:`load the model from Hugging Face <load-models>` and define it as a class variable. The ``HuggingFaceModel`` method provides an efficient mechanism for loading AI models to accelerate model deployment on BentoCloud, reducing image build time and cold start time.

3. The ``@bentoml.service`` decorator also allows you to :doc:`define the runtime environment </build-with-bentoml/runtime-environment>` for a Bento, the unified distribution format in BentoML. A Bento is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

   Here is an example:

   .. code-block:: python
      :caption: `service.py`

      my_image = bentoml.images.PythonImage(python_version='3.11') \
                    .requirements_file("requirements.txt")

      @bentoml.service(
            image=my_image, # Apply the specifications
            ...
      )
      class VLLM:
            ...

4. Use the ``@bentoml.asgi_app`` decorator to mount a FastAPI application, which provides OpenAI-compatible endpoints for chat completions and model listing. The ``path='/v1'`` sets the base path for the API. This allows you to serve the model inference logic alongside the FastAPI application in the same Service. For more information, see :doc:`/build-with-bentoml/asgi`.

   .. code-block:: python
      :caption: `service.py`

      openai_api_app = fastapi.FastAPI()

      @bentoml.asgi_app(openai_api_app, path='/v1')
      @bentoml.service(
          ...
      )
      class VLLM:
          model_id = ENGINE_CONFIG['model']
          model = bentoml.models.HuggingFaceModel(model_id, exclude=['*.pth', '*.pt'])

          def __init__(self) -> None:
              import vllm.entrypoints.openai.api_server as vllm_api_server

              # Define the OpenAI-compatible endpoints
              OPENAI_ENDPOINTS = [
                  ['/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
                  ['/models', vllm_api_server.show_available_models, ['GET']],
              ]

              # Register each endpoint
              for route, endpoint, methods in OPENAI_ENDPOINTS:
                  openai_api_app.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
              ...

5. Use the ``@bentoml.api`` decorator to define an HTTP endpoint ``generate`` for the model inference logic. It will asynchronously stream the responses to the client and perform chat completions using OpenAI-compatible API calls.

   .. code-block:: python
      :caption: `service.py`

      class VLLM:
          ...

          @bentoml.api
          async def generate(
              self,
              prompt: str = 'Who are you? Please respond in pirate speak!',
              max_tokens: typing_extensions.Annotated[
                  int, annotated_types.Ge(128), annotated_types.Le(MAX_TOKENS)
              ] = MAX_TOKENS,
          ) -> typing.AsyncGenerator[str, None]:
              from openai import AsyncOpenAI

              # Create an AsyncOpenAI client to communicate with the model
              client = AsyncOpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy')
              try:
                  # Send the request to OpenAI for chat completion
                  completion = await client.chat.completions.create(
                      model=self.model_id,
                      messages=[dict(role='user', content=[dict(type='text', text=prompt)])],
                      stream=True,
                      max_tokens=max_tokens,
                  )

                  # Stream the results back to the client
                  async for chunk in completion:
                      yield chunk.choices[0].delta.content or ''
              except Exception:
                  # Handle any exceptions by logging the error
                  logger.error(traceback.format_exc())
                  yield 'Internal error found. Check server logs for more information'
                  return

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-8b-instruct>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image, and deploy it anywhere.

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

2. Clone the `BentoVLLM repository <https://github.com/bentoml/BentoVLLM>`_ and deploy the project. We recommend you create a BentoCloud :doc:`secret </scale-with-bentocloud/manage-secrets-and-env-vars>` to store the required environment variable.

   .. code-block:: bash

        git clone https://github.com/bentoml/BentoVLLM.git
        cd BentoVLLM/llama3.1-8b-instruct
        bentoml secret create huggingface HF_TOKEN=<your-api-key>
        bentoml deploy --secret huggingface

3. Once it is up and running on BentoCloud, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/vllm/llama3-1-on-bentocloud.png
		   :alt: Screenshot of Llama 3.1 model in the BentoCloud Playground interface showing the chat interface for testing the deployed model

    .. tab-item:: Python client

       Create a :doc:`BentoML client </build-with-bentoml/clients>` to call the endpoint. Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: python

          import bentoml

          with bentoml.SyncHTTPClient("https://bentovllm-llama-3-1-8-b-instruct-service-pozo-e3c1c7db.mt-guc1.bentoml.ai") as client:
                response_generator = client.generate(
                    prompt="Who are you? Please respond in pirate speak!",
                    max_tokens=1024,
                )
                for response in response_generator:
                    print(response, end='')

    .. tab-item:: OpenAI-compatible endpoints

        Set the ``base_url`` parameter as the BentoML server address in the OpenAI client.

        .. code-block:: python

            from openai import OpenAI

            client = OpenAI(base_url='https://bentovllm-llama-3-1-8-b-instruct-service-pozo-e3c1c7db.mt-guc1.bentoml.ai/v1', api_key='na')

            # Use the following func to get the available models
            # client.models.list()

            chat_completion = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": "Who are you? Please respond in pirate speak!"
                    }
                ],
                stream=True,
            )
            for chunk in chat_completion:
                # Extract and print the content of the model's reply
                print(chunk.choices[0].delta.content or "", end="")

        .. seealso::

            For more information, see the `OpenAI API reference documentation <https://platform.openai.com/docs/api-reference/introduction>`_.

        If your Service is deployed with :ref:`protected endpoints on BentoCloud <scale-with-bentocloud/manage-api-tokens:access protected deployments>`, you need to set the environment variable ``OPENAI_API_KEY`` to your BentoCloud API key first.

        .. code-block:: bash

            export OPENAI_API_KEY={YOUR_BENTOCLOUD_API_TOKEN}

        Make sure you replace the Deployment URL in the above code snippet. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` to retrieve the endpoint URL.

    .. tab-item:: CURL

       .. code-block:: bash

          curl -s -X POST \
            'https://bentovllm-llama-3-1-8-b-instruct-service-pozo-e3c1c7db.mt-guc1.bentoml.ai/generate' \
            -H 'Content-Type: application/json' \
            -d '{
                "max_tokens": 1024,
                "prompt": "Who are you? Please respond in pirate speak!"
            }'

4. To make sure the Deployment automatically scales within a certain replica range, add the scaling flags:

   .. code-block:: bash

      bentoml deploy --secret huggingface --scaling-min 0 --scaling-max 3 # Set your desired count

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

        git clone https://github.com/bentoml/BentoVLLM.git
        cd BentoVLLM/llama3.1-8b-instruct

        # Recommend Python 3.11
        pip install -r requirements.txt
        export HF_TOKEN=<your-hf-token>

2. Serve it locally.

   .. code-block:: bash

        bentoml serve

   .. note::

      To run this project with Llama 3.1 8B Instruct locally, you need an NVIDIA GPU with at least 16G VRAM.

3. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
