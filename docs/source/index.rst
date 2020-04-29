.. BentoML documentation master file, created by
   sphinx-quickstart on Fri Jun 14 11:20:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================

.. image:: _static/img/bentoml.png
    :alt: BentoML
    :target: https://github.com/bentoml/BentoML


BentoML Documenation
====================

`BentoML <https://github.com/bentoml/BentoML>`_
is an open-source platform for **high-performance machine learning model serving**.

What does BentoML do?
---------------------

* Turn your ML model into production API endpoint with just a few lines of code
* Support all major machine learning training frameworks
* End-to-end model serving solution with DevOps best practices baked in
* Model server with Adaptive micro-batching support, bringing the advantage of batch processing to online serving
* Model management for teams, providing CLI access and Web UI dashboard
* Flexible model deployment orchestration with support for Docker, Kubernetes, KFserving, AWS Lambda, SageMaker, Azure and more


Why BentoML
-----------

Getting Machine Learning models into production is hard. Data Scientists are not experts
in building production services and DevOps best practices. The trained models produced
by a Data Science team are hard to test and hard to deploy. This often leads us to a 
time consuming and error-prone workflow, where a pickled model or weights file is handed
over to a software engineering team.

BentoML is an end-to-end solution for model serving, making it possible for Data Science
teams to build production-ready model serving endpoints, with common DevOps best
practices and performance optimizations baked in.

.. image:: _static/img/bentoml-overview.png
    :alt: BentoML Overview


💻 Getting started with BentoML: :ref:`Quickstart Guide <getting-started-page>`.

👩‍💻 Star/Watch/Fork the `BentoML Github Repository <https://github.com/bentoml/BentoML>`_.

👉 Join the `BentoML Slack community <https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg>`_ to hear about the latest development updates.


___________

.. toctree::
   :maxdepth: 2

   quickstart
   concepts
   examples
   guides/index
   deployment/index
   api/index
   cli
   faq
