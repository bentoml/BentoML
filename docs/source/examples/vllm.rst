===================
LLM inference: vLLM
===================

`vLLM <https://github.com/vllm-project/vllm>`_ is a library designed for efficient serving of LLMs, such as gpt-oss, DeepSeek, Qwen, and Llama. It provides high serving throughput and efficient attention key-value memory management using PagedAttention and continuous batching. It supports a variety of inference optimization techniques, including `prefill-decode disaggregation <https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation/>`_, `speculative decoding <https://handbook.modular.com/inference-optimization/speculative-decoding/>`_, and `KV cache offloading <https://handbook.modular.com/inference-optimization/kv-cache-offloading/>`_.

This document demonstrates how to run LLM inference using BentoML and vLLM.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoVLLM" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
        </div>
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-left: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/bentoml-icon.png" alt="BentoML" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="#try-it-out" style="margin-left: 5px; vertical-align: middle;">Serve with BentoML</a>
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

3. Use the ``@bentoml.service`` decorator to define a BentoML :doc:`Service </build-with-bentoml/services>`, where you can customize how the model will be served. The decorator lets you set :doc:`configurations </reference/bentoml/configurations>` like timeout and GPU resources.

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

4. Within the class, :ref:`load the model from Hugging Face <load-models>` and define it as a class variable. ``HuggingFaceModel`` downloads the model from Hugging Face when the Service starts.

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

You can serve `this example project <https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-8b-instruct>`_ locally, or containerize it as an OCI-compliant image and deploy it anywhere.

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

3. Call the OpenAI-compatible endpoints at `http://localhost:3000/v1 <http://localhost:3000/v1>`_ (the default base URL). Set the ``base_url`` parameter as the BentoML server address in the OpenAI client.

   .. code-block:: python

      from openai import OpenAI

      client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

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

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
