.. BentoML documentation master file, created by
   sphinx-quickstart on Fri Jun 14 11:20:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================

.. image:: _static/img/bentoml.png
    :alt: BentoML
    :target: https://github.com/bentoml/BentoML

.. image:: https://static.scarf.sh/a.png?x-pxid=0beb35eb-7742-4dfb-b183-2228e8caf04c


BentoML Documentation
====================

`BentoML <https://github.com/bentoml/BentoML>`_ is an open-source framework for
**machine learning model serving**, aiming to bridge the gap between Data Science and
DevOps.

Data Scientists can easily package their models trained with any ML framework using
BentoMl and reproduce the model for serving in production. BentoML helps with
managing packaged models in the BentoML format, and allows DevOps to deploy them as
online API serving endpoints or offline batch inference jobs, on any cloud platform.


💻 Get started with BentoML: :ref:`Quickstart Guide <getting-started-page>` | `Quickstart on Google Colab <https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`_

👩‍💻 Star/Watch/Fork the `BentoML Github Repository <https://github.com/bentoml/BentoML>`_.

👉 To connect with the community and ask questions, check out
`BentoML Discussions on Github <https://github.com/bentoml/BentoML/discussions>`_ and the
`Bentoml Slack Community <https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg>`_.


What does BentoML do?
---------------------

* Package models trained with any framework and reproduce them for model serving in production
* Package once and deploy anywhere, supporting Docker, Kubernetes, Apache Spark, Airflow, Kubeflow, Knative, AWS Lambda, SageMaker, Azure ML, GCP, Heroku and more
* High-Performance API model server with adaptive micro-batching support
* Central hub for teams to manage and access packaged models via Web UI and API

Why BentoML
-----------

Moving trained Machine Learning models to serving applications in production is hard.
Data Scientists are not experts in building production services. The trained models they
produced are loosely specified and hard to deploy. This often leads ML teams to a
time-consuming and error-prone process, where a jupyter notebook along with pickle and
protobuf file being handed over to ML engineers, for turning the trained model into
services that can be properly deployed and managed by DevOps.

BentoML is framework for ML model serving. It provides high-level APIs for Data
Scientists to create production-ready prediction services, without them worrying about
the infrastructure needs and performance optimizations. BentoML does all those under the
hood, which allows DevOps to seamlessly work with Data Science team, helping to deploy
and operate their models, packaged in the BentoML format.

Check out `Frequently Asked Questions <https://docs.bentoml.org/en/latest/faq.html>`_ 
page on how does BentoML compares to Tensorflow-serving, Clipper, AWS SageMaker, MLFlow,
etc.

.. image:: _static/img/bentoml-overview.png
    :alt: BentoML Overview


___________

.. toctree::
   :maxdepth: 2

   quickstart
   concepts
   examples
   frameworks
   guides/index
   deployment/index
   api/index
   cli
   faq
