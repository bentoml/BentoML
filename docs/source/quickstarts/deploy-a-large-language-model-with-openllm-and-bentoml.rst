======================================================
Deploy a large language model with OpenLLM and BentoML
======================================================

As an important component in the BentoML ecosystem, `OpenLLM <https://github.com/bentoml/OpenLLM>`_ is an open platform designed to facilitate the
operation and deployment of large language models (LLMs) in production. The platform provides functionalities that allow users to fine-tune, serve,
deploy, and monitor LLMs with ease. OpenLLM supports a wide range of state-of-the-art LLMs and model runtimes, such as StableLM, Falcon, Dolly,
Flan-T5, ChatGLM, StarCoder, and more.

With OpenLLM, you can deploy your models to the cloud or on-premises, and build powerful AI applications. It supports the integration of your LLMs
with other models and services such as LangChain, BentoML, and Hugging Face, thereby allowing the creation of complex AI applications.

This quickstart demonstrates how to integrate OpenLLM with BentoML to deploy a large language model. To learn more about OpenLLM,
you can also try the `OpenLLM tutorial in Google Colab: Serving Llama 2 with OpenLLM <https://colab.research.google.com/github/bentoml/OpenLLM/blob/main/examples/openllm-llama2-demo/openllm_llama2_demo.ipynb>`_.

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

Create a BentoML Service
------------------------

Create a ``service.py`` file to define a BentoML :doc:`Service </concepts/service>` and a model :doc:`Runner </concepts/runner>`. As the Service starts, the model defined in it will be downloaded automatically if it does not exist locally.

.. code-block:: python
   :caption: `service.py`

   from __future__ import annotations

   import bentoml
   import openllm

   model = "dolly-v2"

   llm_runner = openllm.Runner(model)

   svc = bentoml.Service(name="llm-dolly-service", runners=[llm_runner])


   @svc.on_startup
   def download(_: bentoml.Context):
       llm_runner.download_model()


   @svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
   async def prompt(input_text: str) -> str:
       answer = await llm_runner.generate.async_run(input_text)
       return answer[0]["generated_text"]

Here is a breakdown of this ``service.py`` file.

- ``model``: The ``model`` variable is assigned the name of the model to be used (``dolly-v2`` in this example). Run ``openllm models`` to view all supported models and their corresponding model IDs. Note that certain models may only support running on GPUs.
- ``openllm.Runner()``: Creates a :doc:`bentoml.Runner </concepts/runner>` instance for the model specified.
- ``bentoml.Service()``: Creates a BentoML Service named ``llm-dolly-service`` and wraps the previously created Runner into the Service.
- ``@svc.on_startup``: Different from the Transformer model quickstart, this tutorial creates an action to be performed when the Service starts using the ``on_startup`` hook in the ``service.py`` file. It calls the ``download_model()`` function to ensure the necessary model and weights are downloaded if they do not exist locally. This makes sure the Service is ready to serve requests when it starts.
- ``@svc.api()``: Defines an API endpoint for the BentoML Service that takes a text input and outputs a text. The endpoint’s functionality is defined in the ``prompt()`` function: it takes in a string of text, runs it through the model to generate an answer, and returns the generated text.

Use ``bentoml serve`` to start the Service.

.. code-block:: bash

   $ bentoml serve service:svc

   2023-07-11T16:17:38+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:svc" can be accessed at http://localhost:3000/metrics.
   2023-07-11T16:17:39+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)

The server is now active at `http://0.0.0.0:3000 <http://0.0.0.0:3000/>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

         curl -X 'POST' \
            'http://0.0.0.0:3000/prompt' \
            -H 'accept: text/plain' \
            -H 'Content-Type: text/plain' \
            -d '$PROMPT' # Replace $PROMPT here with your prompt.

    .. tab-item:: Python

        .. code-block:: bash

         import requests

         response = requests.post(
            "http://0.0.0.0:3000/prompt",
            headers={
               "accept": "text/plain",
               "Content-Type": "text/plain",
            },
            data="$PROMPT", # Replace $PROMPT here with your prompt.
         )

         print(response.text)

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

   Tag                                                                 Module                              Size       Creation Time
   pt-databricks-dolly-v2-3b:f6c9be08f16fe4d3a719bee0a4a7c7415b5c65df  openllm.serialisation.transformers  5.30 GiB   2023-07-11 16:17:26

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
     - pt-databricks-dolly-v2-3b:latest

Run ``bentoml build`` in your project directory to build the Bento.

.. code-block:: bash

   $ bentoml build

   Building BentoML service "llm-dolly-service:oatecjraxktp6nry" from build context "/Users/demo/Documents/openllm-test".
   Packing model "pt-databricks-dolly-v2-3b:f6c9be08f16fe4d3a719bee0a4a7c7415b5c65df"
   Locking PyPI package versions.

   ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
   ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
   ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
   ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
   ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
   ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

   Successfully built Bento(tag="llm-dolly-service:oatecjraxktp6nry").

   Possible next steps:

    * Containerize your Bento with `bentoml containerize`:
       $ bentoml containerize llm-dolly-service:oatecjraxktp6nry

    * Push to BentoCloud with `bentoml push`:
       $ bentoml push llm-dolly-service:oatecjraxktp6nry

Deploy a Bento
--------------

To containerize the Bento with Docker, run:

.. code-block:: bash

   bentoml containerize llm-dolly-service:oatecjraxktp6nry

You can then deploy the Docker image in different environments like Kubernetes. Alternatively, push the Bento to `BentoCloud <https://bentoml.com/cloud>`_ for distributed deployments of your model.
For more information, see :doc:`/bentocloud/how-tos/deploy-bentos`.

See also
--------

- :doc:`/quickstarts/install-bentoml`
- :doc:`/quickstarts/deploy-a-transformer-model-with-bentoml`
- `OpenLLM tutorial in Google Colab: Serving Llama 2 with OpenLLM <https://colab.research.google.com/github/bentoml/OpenLLM/blob/main/examples/openllm-llama2-demo/openllm_llama2_demo.ipynb>`_
