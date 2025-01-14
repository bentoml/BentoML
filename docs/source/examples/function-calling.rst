=======================
Agent: Function calling
=======================

LLM function calling refers to the capability of LLMs to interact with user-defined functions or APIs through natural language prompts. This allows the model to execute specific tasks, retrieve real-time data, or perform calculations beyond its trained knowledge. As a result, the model can provide more accurate and dynamic responses by integrating external resources or executing code in real-time.

This document demonstrates how to build an AI agent capable of calling a user-defined function using Llama 3.1 70B, powered by `LMDeploy <https://github.com/InternLM/lmdeploy>`_ and BentoML.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoFunctionCalling" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
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

The example Python function defined is used for currency conversion and exposed through an API, allowing users to submit queries like the following:

.. code-block:: bash

    {
       "query": "I want to exchange 42 US dollars to Canadian dollars"
    }

The application processes this request and responds by converting USD to CAD using a fictitious exchange rate of 1 to 3.14159.

.. code-block:: bash

    The converted amount of 42 US dollars to Canadian dollars is 131.95.

This example is ready for easy deployment and scaling on BentoCloud. With a single command, you can deploy a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/function-calling/function-calling-playground.gif

Architecture
------------

This example includes two BentoML Services, a Currency Exchange Assistant and an LLM. The LLM Service exposes an OpenAI-compatible API, so that the Exchange Assistant can call the OpenAI client. Here is the general workflow of this example:

.. image:: ../../_static/img/examples/function-calling/function-calling-diagram.png

1. A user submits a query to the Exchange Assistant's Query API, which processes the query and forwards it to the LLM to determine the required function and extract parameters.
2. With the extracted parameters, the Query API invokes the identified Exchange Function, which is responsible for the exchange conversion using the specified parameters.
3. After the Exchange Function computes the results, these are sent back to the LLM. The LLM then uses this data to generate a natural language response, which is returned to the user through the Exchange Assistant.

Code explanations
-----------------

You can find `the source code in GitHub <https://github.com/bentoml/BentoFunctionCalling/>`_. Below is a breakdown of the key code implementations within this project.

service.py
^^^^^^^^^^

The ``service.py`` file outlines the logic of the two required BentoML Services.

1. Begin by specifying the LLM for the project. This example uses `Llama 3.1 70B Instruct AWQ in INT4 <https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4>`_ and you may choose an alternative model as needed.

   .. code-block:: bash

    	MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

2. Create a Python class (``Llama`` in the example) to initialize the model and tokenizer, and use the following decorators to add BentoML functionalities.

   - ``@bentoml.service``: Converts this class into a BentoML Service. You can optionally set :doc:`configurations </reference/bentoml/configurations>` like timeout and GPU resources to use on BentoCloud. We recommend you use an NVIDIA A100 GPU of 80 GB for optimal performance.
   - ``@bentoml.asgi_app``: Mounts an `existing ASGI application <https://github.com/bentoml/BentoFunctionCalling/blob/main/openai_endpoints.py>`_ defined in the ``openai_endpoints.py`` file to this class. It sets the base path to ``/v1``, making it accessible via HTTP requests. The mounted ASGI application provides OpenAI-compatible APIs and can be served side-by-side with the LLM Service. For more information, see :doc:`/build-with-bentoml/asgi`.

   .. code-block:: python

      import bentoml
      from openai_endpoints import openai_api_app

      @bentoml.asgi_app(openai_api_app, path="/v1")
      @bentoml.service(
          traffic={
              "timeout": 300,
          },
          resources={
              "gpu": 1,
              "gpu_type": "nvidia-a100-80gb",
          },
      )
      class Llama:
         def __init__(self) -> None:
         # Logic to initialize the model and tokenizer
         ...

3. Next, use the ``@bentoml.service`` decorator to create another BentoML Service called ``ExchangeAssistant``. Different from the LLM, function calling does not require GPUs and can be run with a single CPU. Running them on separate instances also means you can scale them independently on BentoCloud later.

   Key elements within the ``ExchangeAssistant`` Service:

   - ``bentoml.depends()``: This function calls the ``Llama`` Service as a dependency, which allows ``ExchangeAssistant`` to utilize all its functionalities. For more information, see :doc:`/build-with-bentoml/distributed-services`.
   - Service initialization: Because the ``Llama`` Service provides OpenAI-compatible endpoints, you can use its HTTP client and ``client_url`` to construct an OpenAI client to interact with it.
   - A front-facing API ``/exchange``: Define the endpoint using the ``@bentoml.api`` decorator to handle currency exchange queries.

   .. code-block:: python

      from openai import OpenAI

      @bentoml.service(resources={"cpu": "1"})
      class ExchangeAssistant:
          # Declare dependency on the Llama class
          llm = bentoml.depends(Llama)

	  def __init__(self):
	      # Setup HTTP client to interact with the LLM
	       self.client = OpenAI(
	            base_url=f"{self.llm.client_url}/v1",
	            http_client=self.llm.to_sync.client,
	            api_key="API_TOKEN_NOT_NEEDED"
	      )
              ...

          @bentoml.api
          def exchange(self, query: str = "I want to exchange 42 US dollars to Canadian dollars") -> str:
            # Implementation logic

4. The ``exchange`` method uses the OpenAI client to integrate function calling capabilities with the specified LLM. After parsing the query to determine the necessary function and extracts relevant parameters, it then invokes the identified exchange function to generate the results. For detailed information on OpenAI's function calling client APIs, see `the OpenAI documentation <https://platform.openai.com/docs/guides/function-calling>`_.

   .. code-block:: python

        @bentoml.api
        def exchange(self, query: str = "I want to exchange 42 US dollars to Canadian dollars") -> str:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "convert_currency",
                        "description": "Convert from one currency to another. Result is returned in the 'converted_amount' key.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "from_currency": {"type": "string", "description": "The source currency to convert from, e.g. USD",},
                                "to_currency": {"type": "string", "description": "The target currency to convert to, e.g. CAD",},
                                "amount": {"type": "number", "description": "The amount to be converted"},
                            },
                            "required": [],
                        },
                    },
                }
            ]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
            response_message = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                tools=tools,
            ).choices[0].message
            tool_calls = response_message.tool_calls

5. You can then call the function and add additional functions as needed. Ensure the function definitions in JSON match the corresponding Python function signatures.

   .. code-block:: python

            # Check if there are function calls from the LLM response
            if tool_calls:

                # Map the function name to the actual method
                available_functions = {
                    "convert_currency": self.convert_currency,
                }

                # Append the initial LLM response to messages for complete context
                messages.append(response_message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    # Call the mapped function with parsed arguments
                    function_response = function_to_call(
                        from_currency=function_args.get("from_currency"),
                        to_currency=function_args.get("to_currency"),
                        amount=function_args.get("amount"),
                    )

                    # Append function responses to the message chain
                    messages.append(
                        {
                            "role": "user",
                            "name": function_name,
                            "content": function_response,
                        }
                    )

                # Generate the final response from the LLM incorporating the function responses
                final_response = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                )
                return final_response.choices[0].message.content
            else:
                return "Unable to use the available tools."

bentofile.yaml
^^^^^^^^^^^^^^

This configuration file defines the build options for a :doc:`Bento </reference/bentoml/bento-build-options>`, the unified distribution format in BentoML, which contains source code, Python packages, model references, and environment setup. It helps ensure reproducibility across development and production environments.

Here is an example file:

.. code-block:: yaml

   service: 'service:ExchangeAssistant'
   labels:
     owner: bentoml-team
     stage: demo
   include:
     - '*.py'
   python:
     requirements_txt: './requirements.txt'
     lock_packages: false
   docker:
     python_version: "3.11"

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoFunctionCalling>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image and deploy it anywhere.

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

2. Clone the repository and deploy the project to BentoCloud.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoFunctionCalling.git
      cd BentoFunctionCalling
      bentoml deploy .

3. Once it is up and running on BentoCloud, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/function-calling/function-calling-playground.png

    .. tab-item:: Python client

       .. code-block:: python

          import bentoml

          with bentoml.SyncHTTPClient("<your_deployment_endpoint_url>") as client:
             response_generator = client.exchange(
                   query="I want to exchange 42 US dollars to Canadian dollars"
    		    )
             for response in response_generator:
                  print(response, end='')

    .. tab-item:: CURL

       .. code-block:: bash

          curl -X 'POST' \
            '<your_deployment_endpoint_url>/exchange' \
            -H 'accept: text/plain' \
            -H 'Content-Type: application/json' \
            -d '{
              "query": "I want to exchange 42 US dollars to Canadian dollars"
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

.. important::

   To serve this project locally, you need an Nvidia GPU with sufficient VRAM to run the LLM. We recommend you use an NVIDIA A100 GPU of 80 GB for the included `Llama 3.1 70B Instruct AWQ in INT4 <https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4>`_ for optimal performance.

1. Clone the project repository and install the dependencies.

   .. code-block:: bash

        git clone https://github.com/bentoml/BentoFunctionCalling.git
        cd BentoFunctionCalling

        # Recommend Python 3.11
        pip install -r requirements.txt

2. Serve it locally.

   .. code-block:: bash

        bentoml serve .

3. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
