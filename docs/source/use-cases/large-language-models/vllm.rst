==============
vLLM inference
==============

`vLLM <https://github.com/vllm-project/vllm>`_ is a library designed for efficient serving of large language models (LLMs). It provides high serving throughput and efficient attention key-value memory management using PagedAttention and continuous batching. It seamlessly integrates with a variety of LLMs, such as Llama, OPT, Mixtral, StableLM, and Falcon.

This document demonstrates how to build an LLM application using BentoML and vLLM.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- If you want to test the project locally, you need a Nvidia GPU with least 16G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

Clone the project repository and install all the dependencies.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoVLLM.git
    cd BentoVLLM
    pip install -r requirements.txt && pip install -f -U "pydantic>=2.0"

Create a BentoML Service
------------------------

Define a :doc:`BentoML Service </guides/services>` to customize the serving logic of your lanaguage model, which uses ``vllm`` as the backend option. You can find the following example ``service.py`` file in the cloned repository.

.. note:: 

    This example Service uses the model ``meta-llama/Llama-2-7b-chat-hf``. You can choose any other model supported by vLLM based on your needs. If you are using the same model in the project, you need to obtain access to it on the `Meta website <https://ai.meta.com/resources/models-and-libraries/llama-downloads/>`_ and `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`_.

.. code-block:: python
    :caption: `service.py`

    import bentoml

    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from typing import Optional, AsyncGenerator, List

    MAX_TOKENS = 1024
    SAMPLING_PARAM = SamplingParams(max_tokens=MAX_TOKENS)
    ENGINE_ARGS = AsyncEngineArgs(
        model='meta-llama/Llama-2-7b-chat-hf',
        max_model_len=MAX_TOKENS
    )
    PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    {user_prompt} [/INST] """

    @bentoml.service(
        traffic={
            "timeout": 300,
        },
        resources={
            "gpu": 1,
            "memory": "16Gi",
        },
    )
    class VLLMService:
        def __init__(self) -> None:
            self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
            self.request_id = 0

        @bentoml.api
        async def generate(self, prompt: str = "Explain superconductors like I'm five years old", tokens: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
            prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
            stream = await self.engine.add_request(self.request_id, prompt, SAMPLING_PARAM, prompt_token_ids=tokens)
            self.request_id += 1
            async for request_output in stream:
                yield request_output.outputs[0].text

This script mainly contains the following two parts:

- Constants and templates

  - ``MAX_TOKENS``, ``SAMPLING_PARAM``, and ``ENGINE_ARGS`` are defined to configure the model's behavior. You can change their values as needed.
  - ``PROMPT_TEMPLATE`` is a pre-defined prompt template that provides interaction context and guidelines for the model.
  
- A BentoML Service named ``VLLMService``. The ``@bentoml.service`` decorator is used to define the ``VLLMService`` class as a BentoML Service, specifying timeout, GPU, and memory allocations.

  - The Service initializes an ``AsyncLLMEngine`` with specified engine arguments (``ENGINE_ARGS``). This engine is responsible for processing the language model requests. It also initializes a ``request_id`` to keep track of individual requests.
  - The Service exposes an asynchronous API endpoint ``generate`` that takes a ``prompt`` string and an optional list of ``tokens``. The method submits the request to the language model engine and increments the ``request_id``. It returns a stream of responses from the language model as an asynchronous generator.

Run ``bentoml serve`` in your project directory to start the Service.

.. code-block:: bash

    $ bentoml serve .

    2024-01-29T13:10:50+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLMService" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/generate' \
                -H 'accept: text/event-stream' \
                -H 'Content-Type: application/json' \
                -d '{
                "prompt": "Explain superconductors like I'\''m five years old",
                "tokens": null
            }'

    .. tab-item:: BentoML client

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                response_generator = client.generate(
                    prompt="Explain superconductors like I'm five years old",
                    tokens=None
                )
                for response in response_generator:
                    print(response)

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/use-cases/large-language-models/vllm/service-ui.png

Deploy to production
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for your application. It is used for packaging your application into a Bento. You can use the following example file in the cloned repository (remember to set your Hugging Face token).

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:VLLMService"
    labels:
      owner: bentoml-team
      stage: demo
    include:
    - "*.py"
    python:
      requirements_txt: "./requirements.txt"
    docker:
      setup_script: "./setup.sh"
    envs:
      - name: HF_TOKEN
        value: Null

Make sure you :doc:`have logged in to BentoCloud </bentocloud/how-tos/manage-access-token>`, then run the following command in your project directory to deploy the application to BentoCloud.

.. code-block:: bash

    bentoml deploy .

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

.. note::

   Alternatively, you can use BentoML to generate an :doc:`OCI-compliant image for a more custom deployment </guides/containerization>`.
