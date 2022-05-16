.. image:: _static/img/bentoml-readme-header.jpeg
    :alt: BentoML


Unified Model Serving Framework
===============================

|pypi_status| |downloads| |actions_status| |documentation_status| |join_slack|

What is BentoML?
----------------
BentoML is an open-source framework that simplifies Machine-Learning model deployment and serves your models at production scale.

.. note::
    The BentoML version 1.0 is currently under beta preview release. For our most recent stable release, see the `0.13-LTS documentation <https://docs.bentoml.org/en/v0.13.1/S>`_.


Learn BentoML
-------------

:ref:`Tutorial: Intro to BentoML <tutorial-page>` will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.
You can also run the code in this tutorial from your web browser: `Intro to BentoML on Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb>`_.


:ref:`Main Concepts <concepts-page>` gives a step-by-step tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.


:ref:`ML Frameworks Guide <frameworks-page>` lays out best practices and example usages by the ML framework used for training models.


:ref:`Advanced Topics <advanced-page>` dives into BentoML's internals, architecture and advanced features, including GPU support, inference graph, monitoring, and customizing docker environment etc.


`Gallery Projects <https://github.com/bentoml/gallery>`_ provides example projects that demonstrate BentoML usage in a variety of different scenarios.


:ref:`Deployment Guide <deployment-page>` helps to understand different types of model deployment solutions and what may best fit your use cases


Staying Informed
----------------

The `BentoML Blog <http://modelserving.com>`_ and `@bentomlai <http://twitt
er.com/bentomlai>`_ on Twitter are the official source for
updates from the BentoML team. Anything important, including major releases and announcements, will be posted there. We also frequently
share tutorials, case studies, and community updates there.

To receive release notification, star & watch the `BentoML project on GitHub <https://github.com/bentoml/bentoml>`_. For release
notes and detailed changelog, see the `Releases <https://github.com/bentoml/BentoML/releases>`_ page.


Getting Involved
----------------

BentoML has a thriving open source community where hundreds of ML practitioners gathered and discuss all things MLOps.
`👉 Join us on slack today! <https://l.linklyhq.com/l/ktOX>`_


Something Missing?
------------------

If something is missing in the documentation or if you found some part confusing, please file an issue
`here <https://github.com/bentoml/BentoML/issues/new/choose>`_ with your suggestions for improvement, or tweet at
the `@bentomlai <http://twitter.com/bentomlai>`_ account. We love hearing from you!



.. toctree::
   :hidden:

   tutorial
   concepts/index
   frameworks/index
   advanced/index
   yatai/index
   bentoctl/index
   api/index
   cli

.. spelling::

.. |pypi_status| image:: https://img.shields.io/pypi/v/bentoml.svg?style=flat-square
   :target: https://pypi.org/project/BentoML
.. |downloads| image:: https://pepy.tech/badge/bentoml
   :target: https://pepy.tech/project/bentoml
.. |actions_status| image:: https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg
   :target: https://github.com/bentoml/bentoml/actions
.. |documentation_status| image:: https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square
   :target: https://docs.bentoml.org/
.. |join_slack| image:: https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack&style=flat-square
   :target: https://join.slack.bentoml.org
