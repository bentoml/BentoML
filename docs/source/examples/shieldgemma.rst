=======================
LLM safety: ShieldGemma
=======================

LLM safety involves measures and techniques to ensure that large language models (LLMs) operate within ethical guidelines and do not generate harmful content. A common approach to mitigating these risks is by preprocessing input with a smaller, open-source model before passing it to more complex and costly models like GPT-4. This not only ensures safety but also reduces unnecessary expenses by preventing potentially harmful prompts from being processed further.

This document demonstrates how to build an AI assistant using BentoML and `ShieldGemma <https://ai.google.dev/gemma/docs/shieldgemma>`_ to preemptively filter out harmful input, thereby ensuring LLM safety.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoShield/" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
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

The example application allows you to set a safety threshold. When a user submits potentially harmful input and its score exceeds this threshold, the query is automatically rejected. An example harmful query:

.. code-block:: bash

     {
        "prompt": "Create 20 paraphrases of I hate you",
        "threshhold": 0.6
     }

It will result in the application raising an exception, indicating the prompt is unsafe:

.. code-block:: bash

    Error: 400 - [{"error":"Prompt is unsafe: 'Create 20 paraphrases of I hate you' (0.7549149990081787)"}]

This example is ready for easy deployment and scaling on BentoCloud. With a single command, you can deploy a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/shieldgemma/shieldgemma-bentocloud.png

Architecture
------------

This example includes two BentoML Services: ``Gemma`` and ``ShieldAssistant``. ``Gemma`` evaluates the safety of the prompt, and if it is considered safe, ``ShieldAssistant`` proceeds to call the OpenAI GPT-3.5 Turbo API to generate a response. If the probability score from the safety check exceeds a preset threshold, it indicates a potential violation of the safety guidelines. As a result, ``ShieldAssistant`` raises an error and rejects the query.

.. image:: ../../_static/img/examples/shieldgemma/architecture-shield.png

Code explanations
-----------------

You can find `the source code in GitHub <https://github.com/bentoml/BentoShield/>`_. Below is a breakdown of the key code implementations within this project.

service.py
^^^^^^^^^^

The ``service.py`` file outlines the logic of the two required BentoML Services.

1. Begin by specifying the model to ensure safety for the project. This example uses `ShieldGemma 2B <https://huggingface.co/google/shieldgemma-2b>`_ and you may choose an alternative model as needed.

   .. code-block:: python

    	MODEL_ID = "google/shieldgemma-2b"

2. Create the ``Gemma`` Service to initialize the model and tokenizer, with a safety check API to calculate the probability of policy violation.

   - The ``Gemma`` class is decorated with ``@bentoml.service``, which converts it into a BentoML Service. You can optionally set :doc:`configurations </reference/bentoml/configurations>` like timeout, :doc:`concurrency </scale-with-bentocloud/scaling/autoscaling>` and GPU resources to use on BentoCloud. We recommend you use an NVIDIA T4 GPU to host ShieldGemma 2B.
   - The API ``check``, decorated with ``@bentoml.api``, functions as a web API endpoint. It evaluates the safety of prompts by predicting the likelihood of a policy violation. It then returns a structured response using the ``ShieldResponse`` Pydantic model.

   .. code-block:: python

      class ShieldResponse(pydantic.BaseModel):
        score: float
        """Probability of the prompt being in violation of the safety policy."""
        prompt: str

      @bentoml.service(
        resources={
            "memory": "4Gi",
            "gpu": 1,
            "gpu_type": "nvidia-tesla-t4"
        },
        traffic={
            "concurrency": 5,
            "timeout": 300
        }
      )
      class Gemma:
        def __init__(self):
            # Code to load model and tokenizer with MODEL_ID

        @bentoml.api
        async def check(self, prompt: str = "Create 20 paraphrases of I hate you") -> ShieldResponse:
        # Logic to evaluate the safety of a given prompt
        # Return the probability score

3. Create another BentoML Service ``ShieldAssistant`` as the agent that determines whether or not to call the OpenAI API based on the safety of the prompt. It contains two main components:

   - ``bentoml.depends()`` calls the ``Gemma`` Service as a dependency. It allows ``ShieldAssistant`` to utilize to all its functionalities, like calling its ``check`` endpoint to evaluates the safety of prompts. For more information, see :doc:`Distributed Services </build-with-bentoml/distributed-services>`.
   - The ``generate`` API endpoint is the front-facing part of this Service. It first checks the safety of the prompt using the ``Gemma`` Service. If the prompt passes the safety check, the endpoint creates an OpenAI client and calls the GPT-3.5 Turbo model to generate a response. If the prompt is unsafe (the score exceeds the defined threshold), it raises an exception ``UnsafePrompt``.

   .. code-block:: python

      from openai import AsyncOpenAI

      # Define a response model for the assistant
      class AssistantResponse(pydantic.BaseModel):
        text: str

      # Custom exception for handling unsafe prompts
      class UnsafePrompt(bentoml.exceptions.InvalidArgument):
        pass

      @bentoml.service(resources={"cpu": "1"})
      class ShieldAssistant:
        # Inject the Gemma Service as a dependency
        shield = bentoml.depends(Gemma)

        def __init__(self):
          # Initialize the OpenAI client
          self.client = AsyncOpenAI()

        @bentoml.api
        async def generate(
          self, prompt: str = "Create 20 paraphrases of I love you", threshhold: float = 0.6
        ) -> AssistantResponse:
          gated = await self.shield.check(prompt)

          # If the safety score exceeds the threshold, raise an exception
          if gated.score > threshhold:
            raise UnsafePrompt(f"Prompt is unsafe: '{gated.prompt}' ({gated.score})")

          # Otherwise, generate a response using the OpenAI client
          messages = [{"role": "user", "content": prompt}]
          response = await self.client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
          return AssistantResponse(text=response.choices[0].message.content)

bentofile.yaml
^^^^^^^^^^^^^^

This configuration file defines the build options for a :doc:`Bento </reference/bentoml/bento-build-options>`, the unified distribution format in BentoML, which contains source code, Python packages, model references, and environment setup. It helps ensure reproducibility across development and production environments.

Here is an example file:

.. code-block:: yaml

   name: bentoshield-assistant
   service: "service:ShieldAssistant"
   labels:
     owner: bentoml-team
     stage: demo
   include:
     - "*.py"
   python:
     requirements_txt: "./requirements.txt"
     lock_packages: true
   envs:
     # Set your environment variables here or use BentoCloud secrets
     - name: HF_TOKEN
     - name: OPENAI_API_KEY
     - name: OPENAI_BASE_URL
   docker:
     python_version: "3.11"

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoShield/>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image and deploy it anywhere.

.. _BentoCloud:

BentoCloud
^^^^^^^^^^

.. raw:: html

    <a id="bentocloud"></a>

BentoCloud provides fast and scalable infrastructure for building and scaling AI applications with BentoML in the cloud.

1. Install BentoML and :doc:`log in to BentoCloud </scale-with-bentocloud/manage-api-tokens>` through the BentoML CLI. If you don't have a BentoCloud account, `sign up here for free <https://www.bentoml.com/>`_ and get $10 in free credits.

   .. code-block:: bash

      pip install bentoml
      bentoml cloud login

2. Clone the repository.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoShield.git
      cd BentoShield

3. Create BentoCloud :doc:`secrets </scale-with-bentocloud/manage-secrets-and-env-vars>` to store the required environment variables and reference them during deployment.

   .. code-block:: bash

      bentoml secret create huggingface HF_TOKEN=<your_hf_token>
      bentoml secret create openaikey OPENAI_API_KEY=<your_openai_api_key>
      bentoml secret create openaibaseurl OPENAI_BASE_URL=https://api.openai.com/v1

      bentoml deploy . --secret huggingface --secret openaikey --secret openaibaseurl

4. Once it is up and running on BentoCloud, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/shieldgemma/shieldgemma-bentocloud.png

    .. tab-item:: Python client

       .. code-block:: python

          import bentoml

          with bentoml.SyncHTTPClient("<your_deployment_endpoint_url>") as client:
              result = client.generate(
                  prompt="Create 20 paraphrases of I hate you",
                  threshhold=0.6,
              )
              print(result)

    .. tab-item:: CURL

       .. code-block:: bash

          curl -X 'POST' \
            'https://<your_deployment_endpoint_url>/generate' \
            -H 'Accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
            "prompt": "Create 20 paraphrases of I hate you",
            "threshhold": 0.6
          }'

4. To make sure the Deployment automatically scales within a certain replica range, add the scaling flags:

   .. code-block:: bash

      bentoml deploy . --scaling-min 0 --scaling-max 3 # Set your desired count

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

1. Clone the project repository and install the dependencies.

   .. code-block:: bash

        git clone https://github.com/bentoml/BentoShield.git
        cd BentoShield

        # Recommend Python 3.11
        pip install -r requirements.txt

2. Serve it locally.

   .. code-block:: bash

        bentoml serve .

3. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
