# BentoML Examples 🎨 [![Twitter Follow](https://img.shields.io/twitter/follow/bentomlai?style=social)](https://twitter.com/bentomlai) [![Slack](https://img.shields.io/badge/Slack-Join-4A154B?style=social)](https://l.linklyhq.com/l/ktO8)

BentoML is an open platform for machine learning in production. It simplifies
model packaging and model management, optimizes model serving workloads to run
at production scale, and accelerates the creation, deployment, and monitoring of
prediction services.

The repository contains a collection of example projects demonstrating
[BentoML](https://github.com/bentoml/BentoML) usage and best practices.

👉 [Pop into our Slack community!](https://join.slack.bentoml.org) We're happy
to help with any issue you face or even just to meet you and hear what you're
working on :)

## Index

| Example                                                                                                                                      | Framework           | Model                                | Functionality                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------ | -------------------------------------------------------------------------- |
| [custom_model_runner](https://github.com/bentoml/BentoML/tree/main/examples/custom_model_runner)                                             | PyTorch             | MNIST                                | Custom Model Runner, Prometheus, gRPC                                      |
| [custom_python_model/lda_classifier](https://github.com/bentoml/BentoML/tree/main/examples/custom_python_model/lda_classifier)               | Picklable           | LDA                                  | Custom Python Model                                                        |
| [custom_python_model/simple_pickable_model](https://github.com/bentoml/BentoML/tree/main/examples/custom_python_model/simple_pickable_model) | Picklable           | Python Function                      |                                                                            |
| [custom_runner/nltk_pretrained_model](https://github.com/bentoml/BentoML/tree/main/examples/custom_runner/nltk_pretrained_model)             | Custom              | NLTK                                 | Custom Runner                                                              |
| [custom_runner/torch_hub_yolov5](https://github.com/bentoml/BentoML/tree/main/examples/custom_runner/torch_hub_yolov5)                       | Custom              | YOLOv5                               | Custom Runner, Torch Hub                                                   |
| [custom_web_serving/fastapi_example](https://github.com/bentoml/BentoML/tree/main/examples/custom_web_serving/fastapi_example)               | SKLearn             | Classification                       | FastAPI                                                                    |
| [custom_web_serving/flask_example](https://github.com/bentoml/BentoML/tree/main/examples/custom_web_serving/flask_example)                   | SKLearn             | Classification                       | Flask                                                                      |
| [inference_graph](https://github.com/bentoml/BentoML/tree/main/examples/inference_graph)                                                     | Transformers        | Text Generation, Text Classification | Hugging Face Model Hub, Inference Graph                                    |
| [kfserving](https://github.com/bentoml/BentoML/tree/main/examples/kfserving)                                                                 | SKLearn             | Classification                       | KServe                                                                     |
| [mlflow/keras](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/keras)                                                           | MLflow, Keras       | Sequential                           |                                                                            |
| [mlflow/lightgbm](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/lightgbm)                                                     | MLflow, LightGBM    | Classification                       |                                                                            |
| [mlflow/pytorch](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/pytorch)                                                       | MLflow, PyTorch     | MNIST                                |                                                                            |
| [mlflow/sklearn_autolog](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/sklearn_autolog)                                       | MLflow, SKLearn     | Linear Regression, Pipeline          | MLflow Automatic Logging                                                   |
| [mlflow/sklearn_logistic_regression](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/sklearn_logistic_regression)               | MLflow, SKLearn     | Logistic Regression                  |                                                                            |
| [mlflow/torchscript/IrisClassification](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/torchscript/IrisClassification)         | MLflow, TorchScript | Classfication                        | MLflow Log Model                                                           |
| [mlflow/torchscript/MNIST](https://github.com/bentoml/BentoML/tree/main/examples/mlflow/torchscript/MNIST)                                   | MLflow, PyTorch     | MNIST                                | MLflow Log Model                                                           |
| [monitoring/task_classification](https://github.com/bentoml/BentoML/tree/main/examples/monitoring/task_classification)                       | SKLearn             | Classfication                        | Model Monitoring, Classification Tasks                                     |
| [pydantic_validation](https://github.com/bentoml/BentoML/tree/main/examples/pydantic_validation)                                             | SKLearn             | Classification                       | Pydantic Model, Validation                                                 |
| [pytorch_mnist](https://github.com/bentoml/BentoML/tree/main/examples/pytorch_mnist)                                                         | PyTorch             | MNIST                                |                                                                            |
| [quickstart](https://github.com/bentoml/BentoML/tree/main/examples/quickstart)                                                               | SKLearn             | Classification                       | Notebook                                                                   |
| [sklearn/linear_regression](https://github.com/bentoml/BentoML/tree/main/examples/sklearn/linear_regression)                                 | SKLearn             | Linear Regression                    |                                                                            |
| [sklearn/pipeline](https://github.com/bentoml/BentoML/tree/main/examples/sklearn/pipeline)                                                   | SKLearn             | Pipeline                             |                                                                            |
| [tensorflow2_keras](https://github.com/bentoml/BentoML/tree/main/examples/tensorflow2_keras)                                                 | TensorFlow, Keras   | MNIST                                | Notebook                                                                   |
| [tensorflow2_native](https://github.com/bentoml/BentoML/tree/main/examples/tensorflow2_native)                                               | TensforFlow         | MNIST                                | Notebook                                                                   |
| [xgboost](https://github.com/bentoml/BentoML/tree/main/examples/xgboost)                                                                     | XGBoost             | DMatrix                              |                                                                            |
| [flax/MNIST](https://github.com/bentoml/BentoML/tree/main/examples/flax/MNIST)                                                               | Flax                | MNIST                                | gRPC, Testing                                                              |
| [triton/onnx](https://github.com/bentoml/BentoML/tree/main/examples/triton/onnx)                                                             | ONNX                | YOLOv5                               | Triton Inference Server, gRPC, Python SDK (Containerization, Serve, Build) |
| [triton/pytorch](https://github.com/bentoml/BentoML/tree/main/examples/triton/pytorch)                                                       | Torchscript         | YOLOv5                               | Triton Inference Server, gRPC, Python SDK (Containerization, Serve, Build) |
| [triton/tensorflow](https://github.com/bentoml/BentoML/tree/main/examples/triton/tensorflow)                                                 | Tensorflow          | YOLOv5                               | Triton Inference Server, gRPC, Python SDK (Containerization, Serve, Build) |
| [kubeflow](https://github.com/bentoml/BentoML/tree/main/examples/kubeflow)                                                                   | XGBoost             | Fraud Detection                      | Kubeflow, Notebook                                                         |

## How to contribute

If you have issues running these projects or have suggestions for improvement,
use [Github Issues 🐱](https://github.com/bentoml/BentoML/issues/new)

If you are interested in contributing new projects to this repo, let's talk 🥰 -
Join us on
[Slack](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)
and share your idea in #bentoml-contributors channel

Before you create a Pull Request, make sure:

- Follow the basic structures and naming conventions of other existing example
  projects
- Ensure your project runs with the latest version of BentoML

For legacy version prior to v1.0.0, see the
[0.13-LTS branch](https://github.com/bentoml/gallery/tree/0.13-LTS).
