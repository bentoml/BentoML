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

    This documentation is about deploying online serving workloads, enssentially deploy
    API server that serves prediction calls via HTTP request. For offline serving (or
    batch serving, batch inference), see :doc:`../guides/batch_serving`.


Automated Deployment Management:

.. toctree::
  :glob:
  :maxdepth: 1

  aws_lambda
  aws_sagemaker


Manual Deployment Tutorials:

.. toctree::
  :glob:
  :maxdepth: 1

  clipper
  aws_ecs
  google_cloud_run
  azure_container_instance
  kubernetes



