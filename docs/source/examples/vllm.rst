===================
LLM inference: vLLM
===================

`vLLM <https://github.com/vllm-project/vllm>`_ is a library designed for efficient serving of large language models (LLMs). It provides high serving throughput and efficient attention key-value memory management using PagedAttention and continuous batching. It seamlessly integrates with a variety of LLMs, such as Llama, OPT, Mixtral, StableLM, and Falcon.

This document demonstrates how to build an LLM application using BentoML and vLLM.

All the source code in this tutorial is available in the `BentoVLLM GitHub repository <https://github.com/bentoml/BentoVLLM>`_.

Prerequisites
-------------

- Python 3.9+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/hello-world` first.
- If you want to test the project locally, you need an Nvidia GPU with at least 16G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

Clone the project repository and install all the dependencies.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoVLLM.git
    cd BentoVLLM/mistral-7b-instruct
    pip install -r requirements.txt && pip install -f -U "pydantic>=2.0"

Create a BentoML Service
------------------------

Define a :doc:`BentoML Service </build-with-bentoml/services>` to customize the serving logic of your lanaguage model, which uses ``vllm`` as the backend option. You can find the following example ``service.py`` file in the cloned repository.

.. note::

    This example Service uses the model ``mistralai/Mistral-7B-Instruct-v0.2``, which requires you to `accept relevant conditions to gain access <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`_. You can choose other models in the `BentoVLLM repository <https://github.com/bentoml/BentoVLLM>`_ or any other model supported by vLLM based on your needs.

.. code-block:: python
    :caption: `service.py`

    import uuid
    from typing import AsyncGenerator

    import bentoml
    from annotated_types import Ge, Le
    from typing_extensions import Annotated

    from bentovllm_openai.utils import openai_endpoints


    MAX_TOKENS = 1024
    PROMPT_TEMPLATE = """<s>[INST]
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

    {user_prompt} [/INST] """

    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    @openai_endpoints(model_id=MODEL_ID)
    @bentoml.service(
        name="mistral-7b-instruct-service",
        traffic={
            "timeout": 300,
            "concurrency": 256, # Matches the default max_num_seqs in the VLLM engine
        },
        resources={
            "gpu": 1,
            "gpu_type": "nvidia-l4",
        },
    )
    class VLLM:
        def __init__(self) -> None:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            ENGINE_ARGS = AsyncEngineArgs(
                model=MODEL_ID,
                max_model_len=MAX_TOKENS,
                enable_prefix_caching=True
            )

            self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        @bentoml.api
        async def generate(
            self,
            prompt: str = "Explain superconductors like I'm five years old",
            max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
        ) -> AsyncGenerator[str, None]:
            from vllm import SamplingParams

            SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
            prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
            stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

            cursor = 0
            async for request_output in stream:
                text = request_output.outputs[0].text
                yield text[cursor:]
                cursor = len(text)

This script mainly contains the following two parts:

- Constant and template

  - ``MAX_TOKENS`` defines the maximum number of tokens the model can generate in a single request.
  - ``PROMPT_TEMPLATE`` is a pre-defined prompt template that provides interaction context and guidelines for the model.

- A BentoML Service named ``VLLM``. The ``@bentoml.service`` decorator is used to define the ``VLLM`` class as a BentoML Service, specifying timeout and GPU.

  - The Service initializes an ``AsyncLLMEngine`` object from the ``vllm`` package, with specified engine arguments (``ENGINE_ARGS``). This engine is responsible for processing the language model requests.
  - The Service exposes an asynchronous API endpoint ``generate`` that accepts ``prompt`` and ``max_tokens`` as input. ``max_tokens`` is annotated to ensure it's at least 128 and at most MAX_TOKENS. Inside the method:

    - The prompt is formatted using ``PROMPT_TEMPLATE`` to enforce the model's output to adhere to certain guidelines.
    - ``SamplingParams`` is configured with the ``max_tokens`` parameter, and a request is added to the model's queue using ``self.engine.add_request``. Each request is uniquely identified using a uuid.
    - The method returns an asynchronous generator to stream the model's output as it becomes available.

.. note::

    This Service uses the ``@openai_endpoints`` decorator to set up OpenAI-compatible endpoints (``chat/completions`` and ``completions``). This means your client can interact with the backend Service (in this case, the VLLM class) as if they were communicating directly with OpenAI's API. In addition, it is also possible to generate structured output like JSON using the endpoints.

    This is made possible by this `utility <https://github.com/bentoml/BentoVLLM/tree/main/mistral-7b-instruct/bentovllm_openai>`_, which does not affect your BentoML Service code, and you can use it for other LLMs as well.

    See the **OpenAI-compatible endpoints** tab below for interaction details.

Run ``bentoml serve`` in your project directory to start the Service.

.. code-block:: bash

    $ bentoml serve .

    2024-01-29T13:10:50+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLM" listening on http://localhost:3000 (Press CTRL+C to quit)

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
                "max_tokens": 1024
            }'

    .. tab-item:: Python client

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                response_generator = client.generate(
                    prompt="Explain superconductors like I'm five years old",
                    max_tokens=1024
                )
                for response in response_generator:
                    print(response, end='')

    .. tab-item:: OpenAI-compatible endpoints

        The ``@openai_endpoints`` decorator provides OpenAI-compatible endpoints (``chat/completions`` and ``completions``) for the Service. To interact with them, simply set the ``base_url`` parameter as the BentoML server address in the client.

        .. code-block:: python

            from openai import OpenAI

            client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

            # Use the following func to get the available models
            client.models.list()

            chat_completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[
                    {
                        "role": "user",
                        "content": "Explain superconductors like I'm five years old"
                    }
                ],
                stream=True,
            )
            for chunk in chat_completion:
                # Extract and print the content of the model's reply
                print(chunk.choices[0].delta.content or "", end="")

        .. seealso::

            `OpenAI API reference documentation <https://platform.openai.com/docs/api-reference/introduction>`_

        These OpenAI-compatible endpoints support `vLLM extra parameters <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters>`_. For example, you can force the ``chat/completions`` endpoint to output a JSON object by using ``guided_json``:

        .. code-block:: python

            from openai import OpenAI

            client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

            # Use the following func to get the available models
            client.models.list()

            json_schema = {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }

            chat_completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    }
                ],
                extra_body=dict(guided_json=json_schema),
            )
            print(chat_completion.choices[0].message.content)  # Return something like: {"city": "Paris"}

        If your Service is deployed with :ref:`protected endpoints on BentoCloud <scale-with-bentocloud/manage-api-tokens:access protected deployments>`, you need to set the environment variable ``OPENAI_API_KEY`` to your BentoCloud API key first.

        .. code-block:: bash

            export OPENAI_API_KEY={YOUR_BENTOCLOUD_API_TOKEN}

        You can then use the following line to replace the client in the above code snippet. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` to retrieve the endpoint URL.

        .. code-block:: python

            client = OpenAI(base_url='your_bentocloud_deployment_endpoint_url/v1')

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/examples/vllm/service-ui.png

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

1. First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for your application. It is used for packaging your application into a Bento. Here is an example file in the project:

   .. code-block:: yaml
        :caption: `bentofile.yaml`

        service: "service:VLLM"
        labels:
          owner: bentoml-team
          stage: demo
        include:
          - "*.py"
          - "bentovllm_openai/*.py"
        python:
          requirements_txt: "./requirements.txt"
        envs:
          - name: HF_TOKEN

2. :ref:`Log in to BentoCloud <scale-with-bentocloud/manage-api-tokens:Log in to BentoCloud using the BentoML CLI>`.

   .. code-block:: bash

        bentoml cloud login

3. Create a BentoCloud :doc:`secret </scale-with-bentocloud/manage-secrets-and-env-vars>` to store the required environment variable and reference it during deployment.

   .. code-block:: bash

        bentoml secret create huggingface HF_TOKEN=<your_hf_token>
        bentoml deploy . --secret huggingface

4. Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

   .. image:: ../../_static/img/examples/vllm/vllm-bentocloud.png

   .. note::

       For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
