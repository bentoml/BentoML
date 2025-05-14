================
Agent: LangGraph
================

`LangGraph <https://langchain-ai.github.io/langgraph/>`_ is an open-source library for building stateful, multi-actor applications with LLMs. It allows you to define diverse control flows to create agent and multi-agent workflows.

This document demonstrates how to serve a LangGraph agent application with BentoML.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoLangGraph" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
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

The example LangGraph agent invokes DuckDuckGo to retrieve the latest information when the LLM used lacks the necessary knowledge. For example:

.. code-block:: bash

     {
        "query": "Who won the gold medal at the men's 100 metres event at the 2024 Summer Olympic?"
     }

Example output:

.. code-block:: bash

    Noah Lyles (USA) won the gold medal at the men's 100 metres event at the 2024 Summer Olympic Games. He won by five-thousands of a second over Jamaica's Kishane Thompson.

This example is ready for easy deployment and scaling on BentoCloud. You can use either external LLM APIs or deploy an open-source LLM together with the LangGraph agent. With a single command, you get a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/langgraph/langgraph-agent-on-bentocloud.png
    :alt: Screenshot of LangGraph agent deployed on BentoCloud showing the query interface for asking questions with search capabilities

Architecture
------------

This project consists of two main components: a BentoML Service that serves a LangGraph agent as REST APIs and an LLM that generates text. The LLM can be an external API like Claude 3.5 Sonnet or an open-source model served via BentoML (Ministral-8B-Instruct-2410 in this example).

.. image:: ../../_static/img/examples/langgraph/langgraph-bentoml-architecture.png
    :alt: Architecture diagram showing the LangGraph agent workflow with BentoML, illustrating how user queries flow through the agent to external tools and LLM components

After a user submits a query, it is processed through the LangGraph agent, which includes:

- An ``agent`` `node <https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes>`_ that uses the LLM to understand the query and decide on actions.
- A ``tools`` node that can invoke external tools if needed.

In this example, if the LLM needs additional information, the ``tools`` node calls DuckDuckGo to search the internet for the necessary data. DuckDuckGo then returns the search results to the agent, which compiles the information and delivers the final response to the user.

Code explanations
-----------------

This `example <https://github.com/bentoml/BentoLangGraph>`_ contains the following two sub-projects that demonstrate the use of different LLMs:

- `langgraph-anthropic <https://github.com/bentoml/BentoLangGraph/tree/main/langgraph-anthropic>`_ uses Claude 3.5 Sonnet
- `langgraph-mistral <https://github.com/bentoml/BentoLangGraph/tree/main/langgraph-mistral>`_ uses Ministral-8B-Instruct-2410

Both sub-projects follow the same logic for implementing the LangGraph agent. This document explains the key code implementation in langgraph-mistral.

service.py
^^^^^^^^^^

1. The `service.py <https://github.com/bentoml/BentoLangGraph/blob/main/langgraph-mistral/service.py>`_ file defines a BentoML :doc:`Service </build-with-bentoml/services>` ``LLM`` that serves the Ministral-8B-Instruct-2410 model. You can switch to a different model by changing the ``model`` value if necessary.

   .. code-block:: python
      :caption: `service.py`

      ENGINE_CONFIG = {
        "model": "mistralai/Ministral-8B-Instruct-2410",
        "tokenizer_mode": "mistral",
        "max_model_len": 4096,
        "enable_prefix_caching": False,
      }

      @bentoml.service
      class LLM:
         model_id = ENGINE_CONFIG["model"]
         ...

   ``LLM`` provides OpenAI-compatible APIs and uses vLLM as the inference backend. It is a dependent BentoML Service and can be invoked by the LangGraph agent. For more information on code explanations, see :doc:`/examples/vllm`.

2. Create a another Service ``SearchAgentService`` to wrap around the LangGraph agent. You can optionally set :doc:`configurations </reference/bentoml/configurations>` like :doc:`workers </build-with-bentoml/parallelize-requests>` and :doc:`concurrency </scale-with-bentocloud/scaling/autoscaling>`.

   .. code-block:: python
      :caption: `service.py`

      @bentoml.service(
            workers=2,
            resources={
                "cpu": "2000m"
            },
            traffic={
                "concurrency": 16,
                "external_queue": True
            }
      )
      class SearchAgentService:
          ...

   For deployment on BentoCloud, we recommend you set ``concurrency`` and enable ``external_queue``. Concurrency refers to the number of requests the Service can handle at the same time. With ``external_queue`` enabled, if the application receives more than 16 requests simultaneously, the extra requests are placed in an external queue. They will be processed once the current ones are completed, allowing you to handle traffic spikes without dropping requests.

3. :doc:`Define the runtime environment </build-with-bentoml/runtime-environment>` for a Bento, the unified distribution format in BentoML. A Bento is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

   Here is an example:

   .. code-block:: python
      :caption: `service.py`

      my_image = bentoml.images.Image(python_version='3.11', lock_python_packages=False) \
                       .requirements_file("requirements.txt")

      @bentoml.service(
           image=my_image, # Apply the specifications
           envs=[{"name": "HF_TOKEN"}],
           ...
      )
      class SearchAgentService:
           ...

4. Use the ``bentoml.depends()`` function to invoke the ``LLM`` Service, which allows ``SearchAgentService`` to utilize all its functionalities, such as calling its OpenAI-compatible API endpoints.

   .. code-block:: python
      :caption: `service.py`

      from langchain_openai import ChatOpenAI

      ...
      class SearchAgentService:
      # Call the LLM Service
      llm_service = bentoml.depends(LLM)

      def __init__(self):
          tools = [search]
          self.tools = ToolNode(tools)
          self.model = ChatOpenAI(
              model=LLM.inner.model_id,
              openai_api_key="N/A",
              openai_api_base=f"{self.llm_service.client_url}/v1",
              temperature=0,
              verbose=True,
              http_client=self.llm_service.to_sync.client,
          ).bind_tools(tools)

          # Logic to create LangGraph graph and add nodes & edges
                ...

   Once the LLM Service is injected, use the `ChatOpenAI <https://python.langchain.com/docs/integrations/chat/openai/>`_ API from ``langchain_openai`` to configure an interface to interact with it. Since the ``LLM`` Service provides OpenAI-compatible API endpoints, you can use its HTTP client (``to_sync.client``) and client URL (``client_url``) to easily construct an OpenAI client for interaction.

   After that, define the LangGraph workflow that uses the model. The LangGraph agent will call this model and build its flow with nodes and edges, connecting the outputs of the LLM with the rest of the system. For detailed explanations of implementing LangGraph workflows, see `the LangGraph documentation <https://langchain-ai.github.io/langgraph/#example>`_.

5. Define a BentoML :doc:`task </get-started/async-task-queues>` endpoint ``invoke`` with ``@bentoml.task`` to handle the LangGraph workflow asynchronously. It is a background task that supports long-running operations. This ensures that complex LangGraph workflows involving external tools can complete without timing out.

   After sending the user's query to the LangGraph agent, the task retrieves the final state and provides the results back to the user.

   .. code-block:: python
      :caption: `service.py`

      # Define a task endpoint
      @bentoml.task
      async def invoke(
          self,
          input_query: str = "What is the weather in San Francisco today?",
      ) -> str:
          try:
              # Invoke the LangGraph agent workflow asynchronously
              final_state = await self.app.ainvoke({"messages": [HumanMessage(content=input_query)]})
              # Return the final message from the workflow
              return final_state["messages"][-1].content
          # Handle errors that may occur during model invocation
          except OpenAIError as e:
              logger.error(f"An error occurred: {e}")
              logger.error(traceback.format_exc())
              return "I'm sorry, but I encountered an error while processing your request. Please try again later."

   .. tip::

      We recommend you use a task endpoint for this LangGraph agent application. This is because the LangGraph agent often uses multi-step workflows including querying an LLM and invoking external tools. Such workflows may take longer than the typical HTTP request cycle. If handled synchronously, your application could face request timeouts, especially under high traffic. BentoML task endpoints solve this problem by offloading long-running tasks to the background. You can send a query and check back later for the results, ensuring smooth inference without timeouts.

6. Optionally, add a streaming API to send intermediate results in real time. Use ``@bentoml.api`` to turn the ``stream`` function into an API endpoint and call ``astream_events`` to stream events generated by the LangGraph agent.

   .. code-block:: python
      :caption: `service.py`

      @bentoml.api
      async def stream(
          self,
          input_query: str = "What is the weather in San Francisco today?",
      ) -> typing.AsyncGenerator[str, None]:
          # Loop through the events generated by the LangGraph workflow
          async for event in self.app.astream_events({"messages": [HumanMessage(content=input_query)]}, version="v2"):
              yield str(event) + "\n"

   For more information about the ``astream_events`` API, see `the LangGraph documentation <https://langchain-ai.github.io/langgraph/how-tos/streaming-content/>`_.

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoLangGraph/>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image, and deploy it anywhere.

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

2. Clone the repository and select the desired project to deploy it. We recommend you create a BentoCloud :doc:`secret </scale-with-bentocloud/manage-secrets-and-env-vars>` to store the required environment variable.

   .. code-block:: bash

        git clone https://github.com/bentoml/BentoLangGraph.git

        # Use Ministral-8B-Instruct-2410
        cd BentoLangGraph/langgraph-mistral
        bentoml secret create huggingface HF_TOKEN=$HF_TOKEN
        bentoml deploy --secret huggingface

        # Use Claude 3.5 Sonnet
        cd BentoLangGraph/langgraph-anthropic
        bentoml secret create anthropic ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
        bentoml deploy --secret anthropic

3. Once it is up and running on BentoCloud, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/langgraph/langgraph-agent-on-bentocloud.png
		   :alt: Screenshot of LangGraph agent deployed on BentoCloud showing the query interface for asking questions with search capabilities

    .. tab-item:: Python client

       .. code-block:: python

          import bentoml

          with bentoml.SyncHTTPClient("<your_deployment_endpoint_url>") as client:
              result = client.invoke(
                  input_query="Who won the gold medal at the men's 100 metres event at the 2024 Summer Olympic?",
              )
              print(result)

    .. tab-item:: CURL

       .. code-block:: bash

          curl -s -X POST \
              'https://<your_deployment_endpoint_url>/invoke' \
              -H 'Content-Type: application/json' \
              -d '{
                  "input_query": "Who won the gold medal at the men's 100 metres event at the 2024 Summer Olympic?"
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

        git clone https://github.com/bentoml/BentoLangGraph.git

        # Recommend Python 3.11

        # Use Ministral-8B-Instruct-2410
        cd BentoLangGraph/langgraph-mistral
        pip install -r requirements.txt
        export HF_TOKEN=<your-hf-token>

        # Use Claude 3.5 Sonnet
        cd BentoLangGraph/langgraph-anthropic
        pip install -r requirements.txt
        export ANTHROPIC_API_KEY=<your-anthropic-api-key>

2. Serve it locally.

   .. code-block:: bash

        bentoml serve

   .. note::

      To run this project with Ministral-8B-Instruct-2410 locally, you need an NVIDIA GPU with at least 16G VRAM.

3. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
