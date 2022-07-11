===============================
Unified Model Serving Framework
===============================

|github_stars| |pypi_status| |downloads| |actions_status| |documentation_status| |join_slack|

----

What is BentoML?
----------------
`BentoML <https://github.com/bentoml/BentoML>`_ is an open-source framework for serving
ML models at production scale. Data Scientists and ML Engineers use BentoML to:

* Accelerate and standardize the process of taking ML models to production across teams
* Build reliable, scalable, and high performance model serving systems
* Provide a flexible MLOps platform that grows with your Data Science needs


.. caution::
    The BentoML version 1.0 is currently under beta preview release. For our most recent
    stable release, see the
    `0.13-LTS documentation <https://docs.bentoml.org/en/v0.13.1/>`_.


Learn BentoML
-------------


.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`💻 Tutorial: Intro to BentoML <tutorial>`
        :link: tutorial
        :link-type: doc

        A simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.

    .. grid-item-card:: :doc:`📖 Main Concepts <concepts/index>`
        :link: concepts/index
        :link-type: doc

        A step-by-step tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.

    .. grid-item-card:: :doc:`🧮 ML Framework Specific Guide <frameworks/index>`
        :link: frameworks/index
        :link-type: doc

        Best practices and example usages by the ML framework used for model training.

    .. grid-item-card:: `🎨 Gallery Projects <https://github.com/bentoml/gallery>`_
        :link: https://github.com/bentoml/gallery
        :link-type: url

        Example projects demonstrating BentoML usage in a variety of different scenarios.

    .. grid-item-card:: :doc:`💪 Advanced Guides <guides/index>`
        :link: guides/index
        :link-type: doc

        Dive into BentoML's advanced features, internals, and architecture, including GPU support, inference graph, monitoring, and performance optimization.

    .. grid-item-card:: `💬 BentoML Community <https://l.linklyhq.com/l/ktOX>`_
        :link: https://l.linklyhq.com/l/ktOX
        :link-type: url

        Join us in our Slack community where hundreds of ML practitioners are contributing to the project, helping other users, and discuss all things MLOps.


For MLOps engineers:

.. grid::  1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 0

    .. grid-item-card:: `🦄️ Yatai <https://github.com/bentoml/Yatai>`_
        :link: https://github.com/bentoml/Yatai
        :link-type: url

        Model Deployment at scale on Kubernetes.

    .. grid-item-card:: `🚀 bentoctl <https://github.com/bentoml/bentoctl>`_
        :link: https://github.com/bentoml/bentoctl
        :link-type: url

        Fast model deployment on any cloud platform.


Staying Informed
----------------

The `BentoML Blog <http://modelserving.com>`_ and `@bentomlai <http://twitt
er.com/bentomlai>`_ on Twitter are the official source for
updates from the BentoML team. Anything important, including major releases and announcements, will be posted there. We also frequently
share tutorials, case studies, and community updates there.

To receive release notification, star & watch the `BentoML project on GitHub <https://github.com/bentoml/bentoml>`_. For release
notes and detailed changelog, see the `Releases <https://github.com/bentoml/BentoML/releases>`_ page.

----

Why are we building BentoML?
----------------------------

Model deployment is one of the last and most important stages in the machine learning
life cycle: only by putting a machine learning model into a production environment and
making predictions for end applications, the full potential of ML can be realized.

Sitting at the intersection of data science and engineering, **model deployment
introduces new operational challenges between these teams**. Data scientists, who are
typically responsible for building and training the model, often don’t have the
expertise to bring it into production. At the same time, engineers, who aren’t used to
working with models that require continuous iteration and improvement, find it
challenging to leverage their know-how and common practices (like CI/CD) to deploy them.
As the two teams try to meet halfway to get the model over the finish line,
time-consuming and error-prone workflows can often be the result, slowing down the pace
of progress.

We at BentoML want to **get your ML models shipped in a fast, repeatable, and scalable
way**. BentoML is designed to streamline the handoff to production deployment, making it
easy for developers and data scientists alike to test, deploy, and integrate their
models with other systems.

With BentoML, data scientists can focus primarily on creating and improving their
models, while giving deployment engineers peace of mind that nothing in the deployment
logic is changing and that production service is stable.

----

Getting Involved
----------------

BentoML has a thriving open source community where hundreds of ML practitioners are
contributing to the project, helping other users and discuss all things MLOps.
`👉 Join us on slack today! <https://l.linklyhq.com/l/ktOX>`_


.. toctree::
   :hidden:

   installation
   tutorial
   concepts/index
   frameworks/index
   guides/index
   yatai/index
   bentoctl/index
   reference/index
   faq
   Community <https://l.linklyhq.com/l/ktOX>
   GitHub <https://github.com/bentoml/BentoML>

.. spelling::

.. |pypi_status| image:: https://img.shields.io/pypi/v/bentoml.svg?style=flat-square
   :target: https://pypi.org/project/BentoML
.. |downloads| image:: https://pepy.tech/badge/bentoml?style=flat-square
   :target: https://pepy.tech/project/bentoml
.. |actions_status| image:: https://github.com/bentoml/bentoml/workflows/CI/badge.svg
   :target: https://github.com/bentoml/bentoml/actions
.. |documentation_status| image:: https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square
   :target: https://docs.bentoml.org/
.. |join_slack| image:: https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack&style=flat-square
   :target: https://l.linklyhq.com/l/ktOX
.. |github_stars| image:: https://img.shields.io/github/stars/bentoml/BentoML?color=%23c9378a&label=github&logo=github&style=flat-square
   :target: https://github.com/bentoml/bentoml
