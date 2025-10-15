===================
LLM inference: vLLM
===================

`vLLM <https://github.com/vllm-project/vllm>`_ is a library designed for efficient serving of LLMs, such as gpt-oss, DeepSeek, Qwen, and Llama. It provides high serving throughput and efficient attention key-value memory management using PagedAttention and continuous batching. It supports a variety of inference optimization techniques, including `prefill-decode disaggregation <https://www.bentoml.com/llm/inference-optimization/prefill-decode-disaggregation>`_, `speculative decoding <https://www.bentoml.com/llm/inference-optimization/speculative-decoding>`_, and `KV cache offloading <https://www.bentoml.com/llm/inference-optimization/kv-cache-offloading>`_.

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

You can find `the source code in GitHub <https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-8b-instruct>`_. Below is a breakdown of the key code implementations.

1. Define model and GPU configurations. This example uses Llama-3.1-8B-Instruct, which requires `access from Hugging Face <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_. You can switch to another LLM from the `BentoVLLM repository <https://github.com/bentoml/BentoVLLM>`_ or any other model supported by vLLM.

   .. code-block:: python
      :caption: `service.py`

      import pydantic
      import bentoml

      # Use Pydantic to validate data
      class BentoArgs(pydantic.BaseModel):
        name: str = 'llama3.1-8b-instruct'
        gpu_type: str = 'nvidia-h100-80gb'
        tp: int = 1 # One GPU here for tensor parallelism
        model_id: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        # Other optional fields omitted for brevity

      bento_args = bentoml.use_arguments(BentoArgs)

   Specifications are defined using :doc:`template arguments </build-with-bentoml/template-arguments>`, which allow you to pass dynamic and validated parameters at serve, build, and deploy time. You can reference them just like regular Python variables.

2. Define :doc:`the runtime environment </build-with-bentoml/runtime-environment>` for a Bento, the unified distribution format in BentoML. A Bento is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

   .. code-block:: python
      :caption: `service.py`

      image = (
        bentoml.images.Image(python_version='3.12').system_packages('curl', 'git').requirements_file('requirements.txt')
      )

3. Use the ``@bentoml.service`` decorator to define a BentoML :doc:`Service </build-with-bentoml/services>`, where you can customize how the model will be served. The decorator lets you set :doc:`configurations </reference/bentoml/configurations>` like timeout and GPU resources to use on BentoCloud.

   For some of the fields, you can simply reference the template arguments defined above:

   .. code-block:: python
      :caption: `service.py`

      @bentoml.service(
        name=bento_args.name,
        envs=[
            {'name': 'UV_NO_PROGRESS', 'value': '1'},
            {'name': 'UV_TORCH_BACKEND', 'value': 'cu128'},
            # Env vars here for uv and vllm
        ],
        image=image, # Apply the runtime specs
        traffic={'timeout': 300},
        resources={
            'gpu': bento_args.tp,
            'gpu_type': bento_args.gpu_type
        },
        # More optional fields
      )
      class LLM:

4. Within the class, :ref:`load the model from Hugging Face <load-models>` and define it as a class variable. The ``HuggingFaceModel`` method provides an efficient mechanism for loading AI models to accelerate model deployment on BentoCloud, reducing image build time and cold start time.

   .. code-block:: python
      :caption: `service.py`

      ...
      class LLM:
        hf_model = bentoml.models.HuggingFaceModel(bento_args.model_id, exclude=[".pth", ".pt", "original/**/*"])

5. The Service can run vLLM's built-in HTTP server and exposes OpenAI-compatible endpoints. You can add extra CLI arguments here as needed.

   .. code-block:: python
      :caption: `service.py`

      ...
      class LLM:
        hf_model = hf_model

        def __command__(self) -> list[str]:
          return [
            'vllm', 'serve', self.hf_model,
            # ...extra CLI args (compilation, max length, kv dtype, etc.)
            '--served-model-name', bento_args.model_id,
          ]

That's all you need for the basic setup. If you want to explore advanced options, like FlashAttention, AMD ROCm support, and KV cache configuration, see `the complete source code on GitHub <https://github.com/bentoml/BentoVLLM/blob/main/llama3.1-8b-instruct/service.py>`_. BentoML allows you to fully customize inference code for your use case.

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

3. Once it is up and running on BentoCloud, you can call the OpenAI-compatible endpoints as below:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/vllm/llama3-1-on-bentocloud.png
		   :alt: Screenshot of Llama 3.1 model deployed on BentoCloud showing the chat interface with example prompts and responses

    .. tab-item:: OpenAI-compatible endpoints

        Set the ``base_url`` parameter as the BentoML server address in the OpenAI client.

        .. code-block:: python

            from openai import OpenAI

            client = OpenAI(base_url='https://llama-3-1-8-b-instruct-ckng-d3767914.mt-guc1.bentoml.ai/v1', api_key='na')

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

3. Call the OpenAI-compatible endpoints at `http://localhost:3000/v1 <http://localhost:3000/v1>`_ (the default base URL).

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
