================================
Unified AI Application Framework
================================

|github_stars| |pypi_status| |actions_status| |documentation_status| |join_slack|

----

What is BentoML?
----------------

`BentoML <https://github.com/bentoml/BentoML>`_ is a framework for building **reliable, scalable and cost-efficient AI applications**. It comes with everything you need for model serving, application packaging, and production deployment.


Learn BentoML
-------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`💻 Getting Started <tutorial>`
        :link: tutorial
        :link-type: doc

        A simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.

    .. grid-item-card:: :doc:`📖 Main Concepts <concepts/index>`
        :link: concepts/index
        :link-type: doc

        A step-by-step tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.

    .. grid-item-card:: :doc:`🧮 ML Framework Guides <frameworks/index>`
        :link: frameworks/index
        :link-type: doc

        Best practices and example usages by the ML framework used for building your model.

    .. grid-item-card:: `🎨 Examples <https://bentoml.com/gallery>`_
        :link: https://github.com/bentoml/BentoML/tree/main/examples
        :link-type: url

        Example projects demonstrating BentoML usage in a variety of different scenarios.

    .. grid-item-card:: :doc:`💪 Advanced Guides <guides/index>`
        :link: guides/index
        :link-type: doc

        Dive into BentoML's advanced features, internals, and architecture, including GPU support, inference graph, monitoring, and performance optimization.

    .. grid-item-card:: :doc:`⚙️ Integrations & Ecosystem <integrations/index>`
        :link: integrations/index
        :link-type: doc

        Learn how BentoML works together with other tools and products in the Data/ML ecosystem.

    .. grid-item-card:: :doc:`☁️ BentoCloud <bentocloud/getting-started/index>`
        :link: bentocloud/getting-started/index
        :link-type: doc

        Fully managed platform for deploying and scaling BentoML in the cloud.

    .. grid-item-card:: `💬 BentoML Community <https://l.bentoml.com/join-slack>`_
        :link: https://l.bentoml.com/join-slack
        :link-type: url

        Join us in our Slack community where thousands of AI application developers are contributing to the project and helping each other.


Staying Informed
----------------

The `BentoML Blog <http://bentoml.com/blog>`_ and `@bentomlai <http://twitt
er.com/bentomlai>`_ on Twitter are the official source for
updates from the BentoML team. Anything important, including major releases and announcements, will be posted there. We also frequently
share tutorials, case studies, and community updates there.

To receive release notification, star & watch the `BentoML project on GitHub <https://github.com/bentoml/bentoml>`_.
For release notes and detailed changelog, see the `Releases <https://github.com/bentoml/BentoML/releases>`_ page.





.. toctree::
   :caption: BentoML
   :hidden:

   installation
   tutorial
   concepts/index
   frameworks/index
   guides/index
   integrations/index
   reference/index
   Examples <https://github.com/bentoml/BentoML/tree/main/examples>

.. toctree::
   :caption: BentoCloud
   :hidden:

   bentocloud/getting-started/index
   bentocloud/how-tos/index
   bentocloud/topics/index
   bentocloud/references/index


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
