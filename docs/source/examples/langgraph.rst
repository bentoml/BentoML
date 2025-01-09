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

Architecture
------------

This project consists of two main components: a BentoML Service that serves a LangGraph agent as REST APIs and an LLM that generates text. The LLM can be an external API like Claude 3.5 Sonnet or an open-source model served via BentoML (Mistral 7B in this example).

.. image:: ../../_static/img/examples/langgraph/langgraph-bentoml-architecture.png

After a user submits a query, it is processed through the LangGraph agent, which includes:

- An ``agent`` `node <https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes>`_ that uses the LLM to understand the query and decide on actions.
- A ``tools`` node that can invoke external tools if needed.

In this example, if the LLM needs additional information, the ``tools`` node calls DuckDuckGo to search the internet for the necessary data. DuckDuckGo then returns the search results to the agent, which compiles the information and delivers the final response to the user.

Code explanations
-----------------

This `example <https://github.com/bentoml/BentoLangGraph>`_ contains the following two sub-projects that demonstrate the use of different LLMs:

- `langgraph-anthropic <https://github.com/bentoml/BentoLangGraph/tree/main/langgraph-anthropic>`_ uses Claude 3.5 Sonnet
- `langgraph-mistral <https://github.com/bentoml/BentoLangGraph/tree/main/langgraph-mistral>`_ uses Mistral 7B Instruct

Both sub-projects follow the same logic for implementing the LangGraph agent. This document explains the key code implementation in langgraph-mistral.

mistral.py
^^^^^^^^^^

The `mistral.py <https://github.com/bentoml/BentoLangGraph/blob/main/langgraph-mistral/mistral.py>`_ file defines a BentoML Service ``MistralService`` that serves the Mistral 7B model. You can switch to a different model by changing the ``MODEL_ID`` if necessary.

.. code-block:: python

   MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

``MistralService`` provides OpenAI-compatible APIs and uses vLLM as the inference backend. It is a dependent BentoML Service and can be invoked by the LangGraph agent.

For more information on code explanations, see :doc:`/examples/vllm`.

service.py
^^^^^^^^^^

The ``service.py`` file defines the ``SearchAgentService``, a BentoML Service that wraps around the LangGraph agent and calls the ``MistralService``.

1. Create a Python class and decorate it with ``@bentoml.service``, which transforms it into a BentoML Service. You can optionally set :doc:`configurations </reference/bentoml/configurations>` like :doc:`workers </build-with-bentoml/parallelize-requests>` and :doc:`concurrency </scale-with-bentocloud/scaling/autoscaling>`.

   .. code-block:: python

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

2. Define the logic to call the ``MistralService``. Use the ``bentoml.depends()`` function to invoke it, which allows ``SearchAgentService`` to utilize all its functionalities, such as calling its OpenAI-compatible API endpoints.

   .. code-block:: python

        from mistral import MistralService
        from langchain_openai import ChatOpenAI

        ...
        class SearchAgentService:
            # OpenAI compatible API
            llm_service = bentoml.depends(MistralService)

            def __init__(self):
                openai_api_base = f"{self.llm_service.client_url}/v1"
                self.model = ChatOpenAI(
                    model="mistralai/Mistral-7B-Instruct-v0.3",
                    openai_api_key="N/A",
                    openai_api_base=openai_api_base,
                    temperature=0,
                    verbose=True,
                    http_client=self.llm_service.to_sync.client,
                )

                # Logic to call the model, create LangGraph graph and add nodes & edge
                ...

   Once the Mistral Service is injected, use the `ChatOpenAI <https://python.langchain.com/docs/integrations/chat/openai/>`_ API from ``langchain_openai`` to configure an interface to interact with it. Since the ``MistralService`` provides OpenAI-compatible API endpoints, you can use its HTTP client (``to_sync.client``) and client URL (``client_url``) to easily construct an OpenAI client for interaction.

   After that, define the LangGraph workflow that uses the model. The LangGraph agent will call this model and build its flow with nodes and edges, connecting the outputs of the LLM with the rest of the system. For detailed explanations of implementing LangGraph workflows, see `the LangGraph documentation <https://langchain-ai.github.io/langgraph/#example>`_.

3. Define a BentoML :doc:`task </get-started/async-task-queues>` endpoint ``invoke`` with ``@bentoml.task`` to handle the LangGraph workflow asynchronously. It is a background task that supports long-running operations. This ensures that complex LangGraph workflows involving external tools can complete without timing out.

   After sending the user's query to the LangGraph agent, the task retrieves the final state and provides the results back to the user.

   .. code-block:: python

        # Define a task endpoint
        @bentoml.task
        async def invoke(
            self,
            input_query: str="What is the weather in San Francisco today?",
        ) -> str:
            try:
                # Invoke the LangGraph agent workflow asynchronously
                final_state = await self.app.ainvoke(
                    {"messages": [HumanMessage(content=input_query)]}
                )
                # Return the final message from the workflow
                return final_state["messages"][-1].content
            # Handle errors that may occur during model invocation
            except OpenAIError as e:
                print(f"An error occurred: {e}")
                import traceback
                print(traceback.format_exc())
                return "I'm sorry, but I encountered an error while processing your request. Please try again later."

   .. tip::

      We recommend you use a task endpoint for this LangGraph agent application. This is because the LangGraph agent often uses multi-step workflows including querying an LLM and invoking external tools. Such workflows may take longer than the typical HTTP request cycle. If handled synchronously, your application could face request timeouts, especially under high traffic. BentoML task endpoints solve this problem by offloading long-running tasks to the background. You can send a query and check back later for the results, ensuring smooth inference without timeouts.

4. Optionally, add a streaming API to send intermediate results in real time. Use ``@bentoml.api`` to turn the ``stream`` function into an API endpoint and call ``astream_events`` to stream events generated by the LangGraph agent.

   .. code-block:: python

        @bentoml.api
        async def stream(
            self,
            input_query: str="What is the weather in San Francisco today?",
        ) -> AsyncGenerator[str, None]:
            # Loop through the events generated by the LangGraph workflow
            async for event in self.app.astream_events(
                {"messages": [HumanMessage(content=input_query)]},
                version="v2"
            ):
                # Yield each event and stream it back
                yield str(event) + "\n"

   For more information about the ``astream_events`` API, see `the LangGraph documentation <https://langchain-ai.github.io/langgraph/how-tos/streaming-content/>`_.

bentofile.yaml
^^^^^^^^^^^^^^

This configuration file defines the build options for a :doc:`Bento </reference/bentoml/bento-build-options>`, the unified distribution format in BentoML, which contains source code, Python packages, model references, and environment setup. It helps ensure reproducibility across development and production environments.

Here is an example file for `BentoLangGraph/langgraph-mistral <https://github.com/bentoml/BentoLangGraph/tree/main/langgraph-mistral>`_:

.. code-block:: yaml

    service: "service:SearchAgentService"
    labels:
      author: "bentoml-team"
      project: "langgraph-example"
    include:
      - "*.py"
    python:
      requirements_txt: "./requirements.txt"
      lock_packages: false
    envs:
      # Set HF environment variable here or use BentoCloud secret
      - name: HF_TOKEN
    docker:
      python_version: "3.11"

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoLangGraph/>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image, and deploy it anywhere.

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

2. Clone the repository and select the desired project to deploy it. We recommend you create a BentoCloud :doc:`secret </scale-with-bentocloud/manage-secrets-and-env-vars>` to store the required environment variable.

   .. code-block:: bash

        git clone https://github.com/bentoml/BentoLangGraph.git

        # Use Mistral 7B
        cd BentoLangGraph/langgraph-mistral
        bentoml secret create huggingface HF_TOKEN=$HF_TOKEN
        bentoml deploy . --secret huggingface

        # Use Claude 3.5 Sonnet
        cd BentoLangGraph/langgraph-anthropic
        bentoml secret create anthropic ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
        bentoml deploy . --secret anthropic

3. Once it is up and running on BentoCloud, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/langgraph/langgraph-agent-on-bentocloud.png

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

      bentoml deploy . --secret huggingface --scaling-min 0 --scaling-max 3 # Set your desired count

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

        # Use Mistral 7B
        cd BentoLangGraph/langgraph-mistral
        pip install -r requirements.txt
        export HF_TOKEN=<your-hf-token>

        # Use Claude 3.5 Sonnet
        cd BentoLangGraph/langgraph-anthropic
        pip install -r requirements.txt
        export ANTHROPIC_API_KEY=<your-anthropic-api-key>

2. Serve it locally.

   .. code-block:: bash

        bentoml serve .

   .. note::

      To run this project with Mistral 7B locally, you need an NVIDIA GPU with at least 16G VRAM.

3. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
