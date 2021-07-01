![bentoml-docker](https://github.com/bentoml/BentoML/blob/master/docs/source/_static/img/bentoml.png)
---
## Model Serving Made Easy

[![pypi status](https://img.shields.io/pypi/v/bentoml.svg?style=flat-square)](https://pypi.org/project/BentoML) [![Downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml) [![Actions Status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions) [![Documentation Status](https://readthedocs.org/projects/bentoml/badge/?version=latest&style=flat-square)](https://docs.bentoml.org/) [![join BentoML Slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack&style=flat-square)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

BentoML is a flexible, high-performance framework for serving, managing, and deploying machine learning models.

-   Supports **Multiple ML frameworks**, including Tensorflow, PyTorch, Keras, XGBoost and [more](https://docs.bentoml.org/en/latest/frameworks.html#frameworks-page)
-   **Cloud native deployment** with Docker, Kubernetes, AWS, Azure and [many more](https://docs.bentoml.org/en/latest/deployment/index.html#deployments-page)
-   **High-Performance** online API serving and offline batch serving
-   Web dashboards and APIs for model registry and deployment management

BentoML bridges the gap between Data Science and DevOps. By providing a standard interface for describing a prediction service, BentoML abstracts away how to run model inference efficiently and how model serving workloads can integrate with cloud infrastructures. [See how it works!](https://github.com/bentoml/BentoML#introduction)

💻 Get started with BentoML: [Quickstart Guide](https://docs.bentoml.org/en/latest/quickstart.html#getting-started-page) | [Quickstart on Google Colab](https://colab.research.google.com/github/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb)

👩‍💻 Star/Watch/Fork the [BentoML Github Repository](https://github.com/bentoml/BentoML).

👉 Join the [community Slack](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg) and [discussions on Github](https://github.com/bentoml/BentoML/discussions).

## Announcement

The `latest` tag for `model-server` and `yatai-service` has been deprecated on Docker Hub.

Tags also have new formats, therefore current format will also be deprecated.

With the removal of `latest` tags, the following usecase is **NOT A BUG**:

```shell
» docker pull bentoml/model-server
Error response from daemon: manifest for bentoml/model-server:latest
not found: manifest unknown: manifest unknown
```

## Overview of Images

Three types of images provided in the registry:
- `runtime`: Includes BentoML latest PyPI releases
- `devel`: Nightly releases from `master` branch.

## Notes

In order to run CUDA-enabled images `nvidia-docker2` is required. Refers to [BentoML's GPU Serving guides](https://docs.bentoml.org/en/latest/guides/gpu_serving.html) on how to use BentoML's CUDA images.

### `yatai-service` 0.13.0 tags



