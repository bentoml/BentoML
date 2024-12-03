=====================
BentoML Documentation
=====================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

BentoML is a **Unified Inference Platform** for deploying and scaling AI systems with any model, on any cloud.

Featured examples
-----------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/examples/vllm`
        :link: /examples/vllm
        :link-type: doc

        Deploy an AI application using vLLM as the backend for high-throughput and memory-efficient inference.

    .. grid-item-card:: :doc:`/examples/function-calling`
        :link: /examples/function-calling
        :link-type: doc

        Deploy an AI agent capable of calling user-defined functions.

    .. grid-item-card:: :doc:`/examples/shieldgemma`
        :link: /examples/shieldgemma
        :link-type: doc

        Deploy an AI assistant using ShieldGemma to filter out harmful input before they are processed further.

    .. grid-item-card:: :doc:`/examples/sdxl-turbo`
        :link: /examples/sdxl-turbo
        :link-type: doc

        Deploy an image generation application capable of creating high-quality visuals with just a single inference step.

    .. grid-item-card:: :doc:`/examples/controlnet`
        :link: /examples/controlnet
        :link-type: doc

        Deploy a ControlNet application to influence image composition, adjust specific elements, and ensure spatial consistency.

    .. grid-item-card:: :doc:`More examples 👉</examples/overview>`
        :link: /examples/overview
        :link-type: doc

        Explore more examples to build AI applications with BentoML.

What is BentoML
---------------

BentoML is a **Unified Inference Platform** for deploying and scaling AI models with production-grade reliability, all without the complexity of managing infrastructure. It enables your developers to **build AI systems 10x faster with custom models, scale efficiently in your cloud, and maintain complete control over security and compliance**.

.. image:: ../../_static/img/homepage/bentoml-inference-platform.png

To get started with BentoML:

- Use `pip <https://pip.pypa.io/en/stable/installation/>`_ to install the `BentoML open-source model serving framework <https://github.com/bentoml/BentoML>`_, which is distributed as a Python package on `PyPI <https://pypi.org/project/bentoml/>`_.

  .. code-block:: bash

     # Recommend Python 3.9+
     pip install bentoml

- `Sign up for BentoCloud <https://www.bentoml.com/>`_ with $10 free credits.

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

    .. grid-item-card:: :doc:`/scale-with-bentocloud/deployment/create-deployments`
        :link: /scale-with-bentocloud/deployment/create-deployments
        :link-type: doc

        Deploy your AI application to production with one command.

    .. grid-item-card:: :doc:`/scale-with-bentocloud/scaling/autoscaling`
        :link: /scale-with-bentocloud/scaling/autoscaling
        :link-type: doc

        Configure fast autoscaling to achieve optimal performance.

    .. grid-item-card:: :doc:`/build-with-bentoml/gpu-inference`
        :link: /build-with-bentoml/gpu-inference
        :link-type: doc

        Run model inference on GPUs with BentoML.

    .. grid-item-card:: :doc:`/scale-with-bentocloud/codespaces`
        :link: /scale-with-bentocloud/codespaces
        :link-type: doc

        Develop with powerful cloud GPUs using your favorite IDE.

    .. grid-item-card:: :doc:`/build-with-bentoml/model-loading-and-management`
        :link: /build-with-bentoml/model-loading-and-management
        :link-type: doc

        Load and serve your custom models with BentoML.

Stay informed
-------------

The BentoML team uses the following channels to announce important updates like major product releases and share tutorials, case studies, as well as community news.

- `BentoML Blog <http://bentoml.com/blog>`_
- `BentoML X account <http://twitter.com/bentomlai>`_
- `BentoML LinkedIn account <https://www.linkedin.com/company/bentoml/>`_
- `BentoML Slack community <https://l.bentoml.com/join-slack>`_

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
   get-started/cloud-deployment

.. toctree::
   :caption: Learn by Examples
   :hidden:

   examples/overview
   examples/vllm
   examples/function-calling
   examples/langgraph
   examples/shieldgemma
   examples/rag
   examples/sdxl-turbo
   examples/comfyui
   examples/controlnet
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
   build-with-bentoml/distributed-services
   build-with-bentoml/lifecycle-hooks
   build-with-bentoml/asgi
   build-with-bentoml/gradio
   build-with-bentoml/observability/index
   build-with-bentoml/error-handling
   build-with-bentoml/testing

.. toctree::
   :caption: Scale with BentoCloud
   :hidden:

   scale-with-bentocloud/deployment/index
   scale-with-bentocloud/scaling/index
   scale-with-bentocloud/manage-secrets-and-env-vars
   scale-with-bentocloud/manage-api-tokens
   scale-with-bentocloud/codespaces
   scale-with-bentocloud/administering/index

.. toctree::
   :caption: References
   :hidden:

   reference/bentoml/index
   reference/bentocloud/index

.. |pypi_status| image:: https://img.shields.io/pypi/v/bentoml.svg?style=flat-square
   :target: https://pypi.org/project/BentoML
.. |actions_status| image:: https://github.com/bentoml/bentoml/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/bentoml/bentoml/actions
.. |documentation_status| image:: https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square
   :target: https://docs.bentoml.com/
.. |join_slack| image:: https://badgen.net/badge/Join/Community%20Slack/cyan?icon=slack&style=flat-square
   :target: https://l.bentoml.com/join-slack
.. |github_stars| image:: https://img.shields.io/github/stars/bentoml/BentoML?color=%23c9378a&label=github&logo=github&style=flat-square
   :target: https://github.com/bentoml/bentoml
