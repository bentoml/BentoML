Getting Started
===============

|pypi_status| |downloads| |actions_status| |documentation_status| |join_slack|

This page is an overview of the BentoML documentation and related resources.

BentoML is an open-source framework that simplifies ML model deployment and serves your models at production scale in minutes


The BentoML version 1.0 is around the corner. For stable release version 0.13, see
the `0.13-LTS documentation <https://docs.bentoml.org/en/v0.13.1/S>`_. Version 1.0 is
under active development, you can be of great help by testing out the preview release,
reporting issues, contribute to the documentation and create sample gallery projects.


Try BentoML
-----------

:ref:`Tutorial: Intro to BentoML <tutorial-page>`
will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.

quickstart guide

notebook

Learn BentoML
-------------

:ref:`Main Concepts <concepts-page>` will give a comprehensive tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.


:ref:`ML frameworks <frameworks-page>` lays out best practices and example usages by the ML framework used for training models.


:ref:`Advanced Topics <guides-page>` showcases advanced features in BentoML, including GPU support, inference graph, monitoring, and customizing docker environment etc.



`🦄️ Yatai <https://github.com/bentoml/yatai>`_: Run BentoML workflow at scale on Kubernetes
`🚀 bentoctl <https://github.com/bentoml/bentoctl>`_: Fast model deployment with BentoML on cloud platforms





Staying Informed
----------------

The `BentoML Blog <http://modelserving.com>`_ is the official source for updates from the BentoML team. Anything important,
including major releases and announcements, will be posted there. We also frequently share tutorials, case studies, and
community updates there.

You can also follow `@bentomlai <http://twitter.com/bentomlai>`_ on Twitter.

To receive release notification, star & watch the `BentoML project on GitHub <https://github.com/bentoml/bentoml>`_. For release
notes and detailed changelog, see the `Releases <https://github.com/bentoml/BentoML/releases>`_ page.


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
