.. _deployments-page:

Deployment Guides
=================

BentoML provides a set of APIs and CLI commands for automating cloud deployment workflow
which gets your BentoService API server up and running in the cloud, and allows you to
easily update and monitor the service. Currently BentoML have implemented this workflow
for AWS Lambda and AWS Sagemaker. More platforms such as AWS EC2, Kubernetes Cluster,
Azure Virtual Machines are on our roadmap.

You can also manually deploy the BentoService API Server or its docker image to cloud
platforms, and we've created a few step by step tutorials for doing that.

.. note::

    This documentation is about deploying online serving workloads, essentially deploy
    API server that serves prediction calls via HTTP requests. For offline serving (or
    batch serving, batch inference), see
    :ref:`Model Serving Guide <concepts-model-serving>`.


If you are at a small team with limited DevOps support, BentoML provides a fully
automated deployment management for AWS EC2, AWS Lambda, AWS SageMaker, and Azure
Functions. It provides the basic model deployment functionalities with minimum setup.
Here are the detailed guides for each platform:

.. toctree::
  :glob:
  :maxdepth: 1

  aws_lambda
  aws_sagemaker
  aws_ec2
  azure_functions

BentoML also makes it very easy to deploy your models on any cloud platforms or your
in-house custom infrastructure. Here are deployment guides for popular cloud services
and open source platforms:

.. toctree::
  :glob:
  :maxdepth: 1

  kubernetes
  aws_ecs
  heroku
  google_cloud_run
  azure_container_instance
  knative
  kubeflow
  kfserving
  clipper
  sql_server



