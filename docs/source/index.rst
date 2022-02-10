.. BentoML documentation main file, created by
   sphinx-quickstart on Fri Jun 14 11:20:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: _static/img/bentoml-readme-header.jpeg
    :alt: BentoML
    :target: https://github.com/bentoml/BentoML


Unified Model Serving Framework
===============================

|pypi_status| |downloads| |actions_status| |documentation_status| |join_slack|

BentoML let you create machine learning powered prediction service in minutes and bridges the gap between data science and DevOps.

👉 `Pop into our Slack community! <https://join.slack.bentoml.org>`_ We're happy to help with any issue you face or even just to meet you and hear what you're working on :)

The BentoML version 1.0 is around the corner. For stable release version 0.13, see
the `0.13-LTS branch <https://github.com/bentoml/BentoML/tree/0.13-LTS>`_. Version 1.0 is
under active development, you can be of great help by testing out the preview release,
reporting issues, contribute to the documentation and create sample gallery projects.

Why BentoML
-----------

- The easiest way to turn your ML models into production-ready API endpoints.
- High performance model serving, all in Python.
- Standardlize model packaging and ML service definition to streamline deployment.
- Support all major machine-learning training :ref:`frameworks <frameworks-page>`.
- Deploy and operate ML serving workload at scale on Kubernetes via `Yatai <https://github.com/bentoml/yatai>`_.


Getting Started
---------------
- :ref:`Quickstart guide <getting-started-page>` will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.
- :ref:`Main concepts <concepts-page>` will give a comprehensive tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.
- :ref:`ML frameworks <frameworks-page>` lays out best practices and example usages by the ML framework used for training models.
- :ref:`Advanced guides <guides-page>` showcases advanced features in BentoML, including GPU support, inference graph, monitoring, and customizing docker environment etc.
- Check out other projects from the `BentoML team <https://github.com/bentoml>`_:
  - `🦄️ Yatai <https://github.com/bentoml/yatai>`_: Run BentoML workflow at scale on Kubernetes
  - `🚀 bentoctl <https://github.com/bentoml/bentoctl>`_: Fast model deployment with BentoML on cloud platforms

Community
---------
- To report a bug or suggest a feature request, use `GitHub Issues <https://github.com/bentoml/BentoML/issues/new/choose>`_.
- For other discussions, use `Github Discussions <https://github.com/bentoml/BentoML/discussions>`_.
- To receive release announcements, please join us on `Slack <https://join.slack.bentoml.org>`_.

Contributing
------------
There are many ways to contribute to the project:

- If you have any feedback on the project, share it with the community in `Github Discussions <https://github.com/bentoml/BentoML/discussions>`_ of this project.
- Report issues you're facing and "Thumbs up" on issues and feature requests that are relevant to you.
- Investigate bugs and reviewing other developer's pull requests.
- Contributing code or documentation to the project by submitting a Github pull request. See the `development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_.
- See more in the `contributing guide <ttps://github.com/bentoml/BentoML/blob/main/CONTRIBUTING.md>`_.

Usage Reporting
---------------

BentoML by default collects anonymous usage data using `Amplitude <https://amplitude.com/>`_. 
It only collects BentoML library's own actions and parameters, no user or model data will be collected. 
Here is the `code <https://github.com/bentoml/BentoML/blob/main/bentoml/utils/usage_stats.py>`_ that does it.

This helps the BentoML team to understand how the community is using this tool and what to build next. 
You can easily opt-out of usage tracking by running the BentoML commands with the `--do-not-track` option.

.. code-block:: bash

   > bentoml [command] --do-not-track

You can also opt-out via setting environment variable `BENTOML_DO_NOT_TRACK=True`

.. code-block:: bash

   > export BENTOML_DO_NOT_TRACK=True


License
-------
`Apache License 2.0 <https://github.com/bentoml/BentoML/blob/main/LICENSE>`_

.. toctree::
   :hidden:

   quickstart
   concepts/index
   frameworks/index
   guides/index
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