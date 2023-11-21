======================================================
Deploy a large language model with OpenLLM and BentoML
======================================================

As an important component in the BentoML ecosystem, `OpenLLM <https://github.com/bentoml/OpenLLM>`_ is an open platform designed to facilitate the
operation and deployment of large language models (LLMs) in production. The platform provides functionalities that allow users to fine-tune, serve,
deploy, and monitor LLMs with ease. OpenLLM supports a wide range of state-of-the-art LLMs and model runtimes, such as Llama 2, Mistral, StableLM, Falcon, Dolly,
Flan-T5, ChatGLM, StarCoder, and more.

With OpenLLM, you can deploy your models to the cloud or on-premises, and build powerful AI applications. It supports the integration of your LLMs
with other models and services such as LangChain, LlamaIndex, BentoML, and Hugging Face, thereby allowing the creation of complex AI applications.

This quickstart demonstrates how to integrate OpenLLM with BentoML to deploy a large language model. To learn more about OpenLLM,
you can also try the `OpenLLM tutorial in Google Colab: Serving Llama 2 with OpenLLM <https://colab.research.google.com/github/bentoml/OpenLLM/blob/main/examples/llama2.ipynb>`_.

Prerequisites
-------------

- Make sure you have Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have :doc:`BentoML installed </quickstarts/install-bentoml>`.
- You have a basic understanding of key concepts in BentoML, such as Services and Bentos. We recommend you read :doc:`/quickstarts/deploy-a-transformer-model-with-bentoml` first.
- (Optional) Install `Docker <https://docs.docker.com/get-docker/>`_ if you want to containerize the Bento.
- (Optional) We recommend you create a virtual environment for dependency isolation for this quickstart.
  For more information about virtual environments in Python, see `Creation of virtual environments <https://docs.python.org/3/library/venv.html>`_.

Install OpenLLM
---------------

Run the following command to install OpenLLM.

.. code-block:: bash

   pip install openllm

.. note::

   If you are running on GPUs, we recommend using OpenLLM with vLLM runtime. Install with

   .. code-block:: bash

      pip install "openllm[vllm]"

Create a BentoML Service
------------------------

Create a ``service.py`` file to define a BentoML :doc:`Service </concepts/service>` and a model :doc:`Runner </concepts/runner>`. As the Service starts, the model defined in it will be downloaded automatically if it does not exist locally.

.. code-block:: python
   :caption: `service.py`

    from __future__ import annotations
    import uuid
    from typing import Any, AsyncGenerator, Dict, TypedDict, Union

    from bentoml import Service
    from bentoml.io import JSON, Text
    from openllm import LLM

    llm = LLM[Any, Any]('HuggingFaceH4/zephyr-7b-alpha', backend='vllm')


    svc = Service('tinyllm', runners=[llm.runner])


    class GenerateInput(TypedDict):
      prompt: str
      stream: bool
      sampling_params: Dict[str, Any]


    @svc.api(
      route='/v1/generate',
      input=JSON.from_sample(
        GenerateInput(prompt='What is time?', stream=False, sampling_params={'temperature': 0.73, 'logprobs': 1})
      ),
      output=Text(content_type='text/event-stream'),
    )
    async def generate(request: GenerateInput) -> Union[AsyncGenerator[str, None], str]:
      n = request['sampling_params'].pop('n', 1)
      request_id = f'tinyllm-{uuid.uuid4().hex}'
      previous_texts = [''] * n

      generator = llm.generate_iterator(request['prompt'], request_id=request_id, n=n, **request['sampling_params'])

      async def streamer() -> AsyncGenerator[str, None]:
        async for request_output in generator:
          for output in request_output.outputs:
            i = output.index
            delta_text = output.text[len(previous_texts[i]) :]
            previous_texts[i] += output.text
            yield delta_text

      if request['stream']:
        return streamer()

      final_output = None
      async for request_output in generator:
        final_output = request_output
      assert final_output is not None
      return final_output.outputs[0].text

Here is a breakdown of this ``service.py`` file.

- ``openllm.LLM()``: Creates an LLM abstraction object that allows easy to use APIs for streaming text with optimization built-in. It supports a variety of architectures (See `openllm models` for more information). ``openllm.LLM`` is built on top of a :doc:`bentoml.Runner </concepts/runner>` for this LLM.
- ``bentoml.Service()``: Creates a BentoML Service named ``llm-mistral-service`` and turns the aforementioned `llm.runner` into a `bentoml.Service`.
- ``@svc.api()``: Defines an API endpoint for the BentoML Service that takes a text input and outputs a text. The endpoint’s functionality is defined in the ``generate()`` function: it takes in a string of text, runs it through the model to generate an answer, and returns the generated text. It both supports streaming and one-shot generation.

Use ``bentoml serve`` to start the Service.

.. code-block:: bash

   $ bentoml serve service:svc

   2023-07-11T16:17:38+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:svc" can be accessed at http://localhost:3000/metrics.
   2023-07-11T16:17:39+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)

The server is now active at `http://0.0.0.0:3000 <http://0.0.0.0:3000/>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        For one-shot generation

        .. code-block:: bash


           curl -X 'POST' \
               'http://0.0.0.0:3000/v1/generate' \
               -H 'accept: application/json' \
               -H 'Content-Type: application/json' \
               -d '{"prompt": "What is the meaning of life?", "stream": "False", "sampling_params": {"temperature": 0.73}}'

        For streaming generation

        .. code-block:: bash

           curl -X 'POST' -N \
               'http://0.0.0.0:3000/v1/generate' \
               -H 'accept: application/json' \
               -H 'Content-Type: application/json' \
               -d '{"prompt": "What is the meaning of life?", "stream": "True", "sampling_params": {"temperature": 0.73}}'

    .. tab-item:: Python

        For one-shot generation

        .. code-block:: bash

            import openllm

            client = openllm.HTTPClient('http://localhost:3000')

            response = client.generate("What is the meaning of life?", max_new_tokens=256)

            print(response.outputs[0].text)

        For streaming generation

        .. code-block:: bash

            import openllm

            client = openllm.HTTPClient('http://localhost:3000')

            for it in client.generate_stream("What is the meaning of life?", max_new_tokens=256): print(it.text, flush=True, end='')


    .. tab-item:: Browser

        Visit `http://0.0.0.0:3000 <http://0.0.0.0:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/quickstarts/deploy-a-large-language-model-with-openllm-and-bentoml/service-ui.png

The following example shows the model’s answer to a question about the concept of large language models.

Input:

.. code-block::

   What are Large Language Models?

Output:

.. code-block::

   Large Language Models (LLMs) are statistical models that are trained using a large body of text to recognize words, phrases, sentences, and paragraphs. A neural network is used to train the LLM and a likelihood score is used to quantify the quality of the model’s predictions. LLMs are also called named entity recognition models and can be used in various applications, including question answering, sentiment analysis, and information retrieval.

The model should be downloaded automatically to the Model Store.

.. code-block:: bash

   $ bentoml models list

      Tag                                                                           Module                              Size        Creation Time
      vllm-huggingfaceh4--zephyr-7b-alpha:8af01af3d4f9dc9b962447180d6d0f8c5315da86   openllm.serialisation.transformers  13.49 GiB   2023-11-16 06:32:45

Build a Bento
-------------

After the Service is ready, you can package it into a :doc:`Bento </concepts/bento>` by specifying a configuration YAML file (``bentofile.yaml``) that defines the build options. See :ref:`Bento build options <concepts/bento:Bento build options>` to learn more.

.. code-block:: yaml
   :caption: `bentofile.yaml`

   service: "service:svc"
   include:
   - "*.py"
   python:
      packages:
      - openllm
   models:
     - vllm-huggingfaceh4--zephyr-7b-alpha:latest

Run ``bentoml build`` in your project directory to build the Bento.

.. code-block:: bash

   $ bentoml build

   Building BentoML service "llm-mistral-service:oatecjraxktp6nry" from build context "/Users/demo/Documents/openllm-test".
   Packing model "vllm-huggingfaceh4--zephyr-7b-alpha:8af01af3d4f9dc9b962447180d6d0f8c5315da86"
   Locking PyPI package versions.

   ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
   ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
   ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
   ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
   ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
   ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

   Successfully built Bento(tag="llm-mistral-service:oatecjraxktp6nry").

   Possible next steps:

    * Containerize your Bento with `bentoml containerize`:
       $ bentoml containerize llm-mistral-service:oatecjraxktp6nry

    * Push to BentoCloud with `bentoml push`:
       $ bentoml push llm-mistral-service:oatecjraxktp6nry

Deploy a Bento
--------------

To containerize the Bento with Docker, run:

.. code-block:: bash

   bentoml containerize llm-mistral-service:oatecjraxktp6nry

You can then deploy the Docker image in different environments like Kubernetes. Alternatively, push the Bento to `BentoCloud <https://bentoml.com/cloud>`_ for distributed deployments of your model.
For more information, see :doc:`/bentocloud/how-tos/deploy-bentos`.

See also
--------

- :doc:`/quickstarts/install-bentoml`
- :doc:`/quickstarts/deploy-a-transformer-model-with-bentoml`
- `OpenLLM tutorial in Google Colab: Serving Llama 2 with OpenLLM <https://colab.research.google.com/github/bentoml/OpenLLM/blob/main/examples/openllm-llama2-demo/openllm_llama2_demo.ipynb>`_
