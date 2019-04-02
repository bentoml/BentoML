# BentoML

![project status](https://www.repostatus.org/badges/latest/wip.svg)
![pypi status](https://img.shields.io/pypi/v/bentoml.svg)
![python versions](https://img.shields.io/pypi/pyversions/bentoml.svg)
![build_status](https://travis-ci.org/bentoml/BentoML.svg?branch=master)

BentoML is open source tool for packaging machine learning models and their preprocessing code into container image or python library that can be easily used for testing and production deployment.

* Best Practice Built-in - BentoML has a built-in model server supporting telemetrics and logging, making it easy to integrate with production systems. It tries to achieve best performance possible by enabling dynamic batching, caching, paralyzing preprocessing steps and customizable inference runtime.

* Multiple framework support - BentoML supports a wide range of ML frameworks out-of-the-box including Tensorflow, PyTorch, Scikit-Learn, xgboost and can be easily extended to work with new or custom frameworks.

* Streamlines deployment workflows - BentoML has built-in support for easily deploying models as REST API running on Kubernetes, AWS EC2, ECS, Google Cloud Platform, AWS SageMaker, and Azure ML.
