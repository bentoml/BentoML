===============================
Unified Model Serving Framework
===============================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

`BentoML <https://github.com/bentoml/BentoML>`_ is a Python library for building online serving systems optimized for AI applications and model inference.

Featured examples
-----------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/use-cases/large-language-models/vllm`
        :link: /use-cases/large-language-models/vllm
        :link-type: doc

        Deploy an LLM application using vLLM as the backend for high-throughput and memory-efficient inference.

    .. grid-item-card:: :doc:`/use-cases/diffusion-models/controlnet`
        :link: /use-cases/diffusion-models/controlnet
        :link-type: doc

        Deploy a ControlNet application to influence image composition, adjust specific elements, and ensure spatial consistency.

    .. grid-item-card:: :doc:`/use-cases/large-language-models/function-calling`
        :link: /use-cases/large-language-models/function-calling
        :link-type: doc

        Deploy an AI agent capable of calling user-defined functions.

    .. grid-item-card:: :doc:`/use-cases/large-language-models/shieldgemma`
        :link: /use-cases/large-language-models/shieldgemma
        :link-type: doc

        Deploy an AI assistant using ShieldGemma to filter out harmful input before they are processed further.

    .. grid-item-card:: :doc:`/use-cases/diffusion-models/sdxl-turbo`
        :link: /use-cases/diffusion-models/sdxl-turbo
        :link-type: doc

        Deploy an image generation application capable of creating high-quality visuals with just a single inference step.

    .. grid-item-card:: :doc:`/use-cases/audio/whisperx`
        :link: /use-cases/audio/whisperx
        :link-type: doc

        Deploy a speech recognition application.

Start your BentoML journey
--------------------------

The BentoML documentation provides detailed guidance on the project with hands-on tutorials and examples. If you are a first-time user of BentoML, we recommend that you read the following documents in order:

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`Get started <get-started/index>`
        :link: get-started/index
        :link-type: doc

        Gain a basic understanding of the BentoML open-source framework, its workflow, installation, and a quickstart example.

    .. grid-item-card:: :doc:`Use cases <use-cases/index>`
        :link: use-cases/index
        :link-type: doc

        Create different BentoML projects for common machine learning scenarios, like large language models, image generation, embeddings, speech recognition, and more.

    .. grid-item-card:: :doc:`Guides <guides/index>`
        :link: guides/index
        :link-type: doc

        Dive into BentoML's features and advanced use cases, including GPU support, clients, monitoring, and performance optimization.

    .. grid-item-card:: :doc:`BentoCloud <bentocloud/get-started/>`
        :link: bentocloud/get-started/
        :link-type: doc

        A fully managed platform for deploying and scaling BentoML in the cloud.

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
   :caption: BentoML
   :hidden:

   get-started/index
   use-cases/index
   guides/index
   reference/index

.. toctree::
   :caption: BentoCloud
   :hidden:

   bentocloud/get-started
   bentocloud/how-tos/index
   bentocloud/best-practices/index
   bentocloud/reference/index


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
