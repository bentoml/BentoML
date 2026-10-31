=====================
BentoML Documentation
=====================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

BentoML is a **Unified Model Serving Framework** for building model inference APIs and multi-model serving systems with any open-source or custom AI models.

Featured examples
-----------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`Deploy an open-source LLM endpoint </examples/vllm>`
        :link: /examples/vllm
        :link-type: doc

        Serve large language models with OpenAI-compatible APIs and vLLM inference backend.

    .. grid-item-card:: :doc:`Serve a LangGraph agent </examples/langgraph>`
        :link: /examples/langgraph
        :link-type: doc

        Serve a LangGraph agent that searches the web and calls an external or open-source LLM.

    .. grid-item-card:: :doc:`Serve MLflow models </examples/mlflow>`
        :link: /examples/mlflow
        :link-type: doc

        Import an MLflow model into BentoML and serve it as an inference API.

    .. grid-item-card:: :doc:`Serve XGBoost models </examples/xgboost>`
        :link: /examples/xgboost
        :link-type: doc

        Serve a gradient boosting model for tabular predictions, such as breast cancer classification.

What is BentoML
---------------

BentoML is an open-source Python library for building online serving systems optimized for AI apps and model inference. With BentoML, you can:

- **Build APIs for any AI/ML model.** Turn any model inference script into a REST API server with a few lines of code and standard Python type hints.
- **Package and containerize with ease.** Manage environments, dependencies and model versions in one place. BentoML generates reproducible Docker images that you can deploy to any environment.
- **Maximize CPU/GPU utilization.** Use built-in serving optimizations like adaptive batching, model parallelism, multi-stage pipelines and multi-model composition.
- **Customize everything.** Implement your own APIs or task queues with custom business logic, model inference and multi-model composition. BentoML supports any ML framework, modality and inference runtime.
- **Run in production on your own infrastructure.** Develop, run and debug locally, then deploy with Docker containers or Kubernetes.

To get started, use `pip <https://pip.pypa.io/en/stable/installation/>`_ to install `BentoML <https://github.com/bentoml/BentoML>`_, which is distributed as a Python package on `PyPI <https://pypi.org/project/bentoml/>`_.

.. code-block:: bash

   # Recommend Python 3.9+
   pip install bentoml

How-tos
-------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/build-with-bentoml/services`
        :link: /build-with-bentoml/services
        :link-type: doc

        Build your custom AI APIs with BentoML.

    .. grid-item-card:: :doc:`/get-started/packaging-for-deployment`
        :link: /get-started/packaging-for-deployment
        :link-type: doc

        Package your AI application as a Bento and containerize it for deployment.

    .. grid-item-card:: :doc:`/get-started/agent-skills`
        :link: /get-started/agent-skills
        :link-type: doc

        Let a coding agent deploy your BentoML Service to Kubernetes or EC2.

    .. grid-item-card:: :doc:`/build-with-bentoml/gpu-inference`
        :link: /build-with-bentoml/gpu-inference
        :link-type: doc

        Run model inference on GPUs with BentoML.

    .. grid-item-card:: :doc:`/get-started/adaptive-batching`
        :link: /get-started/adaptive-batching
        :link-type: doc

        Batch requests dynamically for higher throughput and better resource utilization.

    .. grid-item-card:: :doc:`/build-with-bentoml/model-loading-and-management`
        :link: /build-with-bentoml/model-loading-and-management
        :link-type: doc

        Load and serve your custom models with BentoML.

Stay informed
-------------

The BentoML team uses the following channels to announce important updates like major product releases and share tutorials, case studies, as well as community news.

- `BentoML Blog <http://bentoml.com/blog>`_
- `BentoML X account <https://x.com/bentomlai>`_
- `BentoML LinkedIn account <https://www.linkedin.com/company/bentoml/>`_
- `BentoML community forum <https://forum.modular.com/c/bento/31>`_

To receive release notifications, star and watch the `BentoML project on GitHub <https://github.com/bentoml/bentoml>`_.
For release notes and detailed changelogs, see the `Releases <https://github.com/bentoml/BentoML/releases>`_ page.

.. toctree::
   :caption: Get Started
   :hidden:

   get-started/hello-world
   get-started/adaptive-batching
   get-started/model-composition
   get-started/async-task-queues
   get-started/packaging-for-deployment
   get-started/agent-skills

.. toctree::
   :caption: Learn by Examples
   :hidden:

   examples/vllm
   examples/langgraph
   examples/mlflow
   examples/xgboost

.. toctree::
   :caption: Build with BentoML
   :hidden:

   build-with-bentoml/services
   build-with-bentoml/iotypes
   build-with-bentoml/model-loading-and-management
   build-with-bentoml/gpu-inference
   build-with-bentoml/clients
   build-with-bentoml/parallelize-requests
   build-with-bentoml/runtime-environment
   build-with-bentoml/distributed-services
   build-with-bentoml/template-arguments
   build-with-bentoml/lifecycle-hooks
   build-with-bentoml/asgi
   build-with-bentoml/streaming
   build-with-bentoml/websocket
   build-with-bentoml/gradio
   build-with-bentoml/observability/index
   build-with-bentoml/error-handling
   build-with-bentoml/testing

.. toctree::
   :caption: References
   :hidden:

   reference/bentoml/stores
   reference/bentoml/sdk
   reference/bentoml/bento-build-options
   reference/bentoml/cli
   reference/bentoml/client
   reference/bentoml/frameworks/index
   reference/bentoml/configurations
   reference/bentoml/batch
   reference/bentoml/exceptions
   reference/bentoml/container
   reference/bentoml/types

.. |pypi_status| image:: https://img.shields.io/pypi/v/bentoml.svg?style=flat-square
   :target: https://pypi.org/project/BentoML
.. |actions_status| image:: https://github.com/bentoml/bentoml/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/bentoml/bentoml/actions
.. |documentation_status| image:: https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square
   :target: https://docs.bentoml.com/
.. |join_slack| image:: https://badgen.net/badge/Join/Community%20Forum/cyan?style=flat-square
   :target: https://forum.modular.com/c/bento/31
.. |github_stars| image:: https://img.shields.io/github/stars/bentoml/BentoML?color=%23c9378a&label=github&logo=github&style=flat-square
   :target: https://github.com/bentoml/bentoml
