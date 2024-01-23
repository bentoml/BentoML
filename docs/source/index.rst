================================
Unified AI Application Framework
================================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

`BentoML <https://github.com/bentoml/BentoML>`_ is a framework for building **reliable, scalable and cost-efficient AI applications**. It comes with everything you need for model serving, application packaging, and production deployment.

Featured use cases
------------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/use-cases/diffusion-models/sdxl-turbo`
        :link: /use-cases/diffusion-models/sdxl-turbo
        :link-type: doc

        Deploy an image generation application capable of creating high-quality visuals with just a single inference step.

    .. grid-item-card:: :doc:`/use-cases/embeddings/clip-embeddings`
        :link: /use-cases/embeddings/clip-embeddings
        :link-type: doc

        Deploy a CLIP application to convert images and text into embeddings.

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

    .. grid-item-card:: :doc:`BentoCloud <bentocloud/getting-started/index>`
        :link: bentocloud/getting-started/index
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
   Examples <https://github.com/bentoml/BentoML/tree/main/examples>

.. toctree::
   :caption: BentoCloud
   :hidden:

   bentocloud/getting-started/index
   bentocloud/how-tos/index
   bentocloud/topics/index
   bentocloud/best-practices/index
   bentocloud/reference/index


.. |pypi_status| image:: https://img.shields.io/pypi/v/bentoml.svg?style=flat-square
   :target: https://pypi.org/project/BentoML
.. |actions_status| image:: https://github.com/bentoml/bentoml/workflows/CI/badge.svg
   :target: https://github.com/bentoml/bentoml/actions
.. |documentation_status| image:: https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square
   :target: https://docs.bentoml.com/
.. |join_slack| image:: https://badgen.net/badge/Join/Community%20Slack/cyan?icon=slack&style=flat-square
   :target: https://l.bentoml.com/join-slack
.. |github_stars| image:: https://img.shields.io/github/stars/bentoml/BentoML?color=%23c9378a&label=github&logo=github&style=flat-square
   :target: https://github.com/bentoml/bentoml
