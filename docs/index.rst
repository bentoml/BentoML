.. BentoML documentation master file, created by
   sphinx-quickstart on Fri Jun 14 11:20:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BentoML Documentation
===================================

BentoML is a flexible framework that accelerates the workflow of serving and deploying 
machine learning models in the cloud.

It provides two set of high-level APIs:

* BentoService: Turn your trained ML model into versioned file bundle that can be
  deployed as containerize REST API server, PyPI package, CLI tool, or
  batch/streaming job

* YataiService: Manage and deploy your saved BentoML bundles into prediction
  services on Kubernetes cluster or cloud platforms such as AWS Lambda, SageMaker,
  Azure ML, and GCP Function etc

Content
----------
.. toctree::
   :maxdepth: 4

   quickstart
   api/index
   cli
   deployments
   bento_archive
