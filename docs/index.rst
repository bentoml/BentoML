.. BentoML documentation master file, created by
   sphinx-quickstart on Fri Jun 14 11:20:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BentoML Documentation
===================================

BentoML is an open framework for building, shipping and running machine
learning services.

BentoML provides high level APIs for defining machine learning service
and packaging its artifacts, source code, dependencies and configurations
into a production-system-friendly format that is ready for deployment.

Feature Highlights
------------------

* **Multiple Distribution Formats** - Easily package machine learning
  models and preprocessing code into a format that works best with your
  inference scenarios:

  * Docker Image - Deploy as container running REST API server
  * PyPi Package - Integrate into python applications seamlessly
  * CLI tool - Incorporate model into Airflow DAG or CI/CD pipeline
  * Spark UDF - Run batch inference on a large dataset with Spark
  * Serverless Function - Host model on serverless platforms such as AWS Lambda

* **Multiple Frameworks Support** - BentoML supports a wild range of machine
  learning frameworks out-of-box including Tensorflow, PyTorch, Scikit-Learn,
  H2o, XgBoost and can be easily extended to work with new or custom
  frameworks.

* **Deploy Anywhere** - BentoML bundled machine learning service can be easily
  deployed with platforms such as Docker, Kubernetes, Serverless, Airflow and
  Clipper, on cloud providers including AWS, Google Cloud, and Azure.

* **Custom Runtime Backend** - Easily integrate python preprocessing code with
  high performance deep learning runtime backend such as Tensorflow-serving.


Content
----------
.. toctree::
   :maxdepth: 4

   quickstart
   api/index
   cli
   deployments
   bento_archive
