## BentoML

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

The `latest` tag for `model-server` has been deprecated on Docker Hub.

Tags also have new formats, therefore current format will also be deprecated.

With the removal of `latest` tags, the following usecase is **NOT A BUG**:

```shell
» docker pull bentoml/model-server
Using default tag: latest
Error response from daemon: manifest for bentoml/model-server:latest
not found: manifest unknown: manifest unknown
```

## Overview of Images Types

- `runtime`: Includes BentoML latest PyPI releases for `model-server`.
- `devel`: Nightly releases from `master` branch.
- `cudnn`: Builds on top of `runtime` with the addition of [CUDA](https://developer.nvidia.com/gpu-accelerated-libraries) and [CUDNN](https://developer.nvidia.com/cudnn) libraries.

## Notes

In order to run CUDA-enabled images `nvidia-docker2` is required. Refers to [BentoML's GPU Serving guides](https://docs.bentoml.org/en/latest/guides/gpu_serving.html) on how to use BentoML's CUDA images.

## Latest tags for `model-server 0.13.0`

### Debian10

- [`0.13.0-python3.6-slim-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/cudnn/Dockerfile)
- [`0.13.0-python3.6-slim-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/runtime/Dockerfile)
- [`0.13.0-python3.7-slim-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/cudnn/Dockerfile)
- [`0.13.0-python3.7-slim-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/runtime/Dockerfile)
- [`0.13.0-python3.8-slim-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/cudnn/Dockerfile)
- [`0.13.0-python3.8-slim-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/runtime/Dockerfile)
- [`devel-python3.6-slim`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/devel/Dockerfile)
- [`devel-python3.7-slim`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/devel/Dockerfile)
- [`devel-python3.8-slim`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/debian10/devel/Dockerfile)

### Centos8

*WARNING*: POSSIBLE MISSING IMAGE TAGS

Centos upstream images often fail security scans, thus there might be some images missing. Please refers to [Issues section](https://github.com/bentoml/BentoML/issues) for security notices.

- [`0.13.0-python3.6-centos8-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/cudnn/Dockerfile)
- [`0.13.0-python3.6-centos8-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/runtime/Dockerfile)
- [`0.13.0-python3.7-centos8-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/cudnn/Dockerfile)
- [`0.13.0-python3.7-centos8-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/runtime/Dockerfile)
- [`0.13.0-python3.8-centos8-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/cudnn/Dockerfile)
- [`0.13.0-python3.8-centos8-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/runtime/Dockerfile)
- [`devel-python3.6-centos8`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/devel/Dockerfile)
- [`devel-python3.7-centos8`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/devel/Dockerfile)
- [`devel-python3.8-centos8`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos8/devel/Dockerfile)

### Centos7

*WARNING*: POSSIBLE MISSING IMAGE TAGS

Centos upstream images often fail security scans, thus there might be some images missing. Please refers to [Issues section](https://github.com/bentoml/BentoML/issues) for security notices.

- [`0.13.0-python3.6-centos7-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/cudnn/Dockerfile)
- [`0.13.0-python3.6-centos7-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/runtime/Dockerfile)
- [`0.13.0-python3.7-centos7-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/cudnn/Dockerfile)
- [`0.13.0-python3.7-centos7-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/runtime/Dockerfile)
- [`0.13.0-python3.8-centos7-cudnn`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/cudnn/Dockerfile)
- [`0.13.0-python3.8-centos7-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/runtime/Dockerfile)
- [`devel-python3.6-centos7`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/devel/Dockerfile)
- [`devel-python3.7-centos7`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/devel/Dockerfile)
- [`devel-python3.8-centos7`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/centos7/devel/Dockerfile)

### Amazonlinux2

- [`0.13.0-python3.6-ami2-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/amazonlinux2/runtime/Dockerfile)
- [`0.13.0-python3.7-ami2-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/amazonlinux2/runtime/Dockerfile)
- [`0.13.0-python3.8-ami2-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/amazonlinux2/runtime/Dockerfile)

### Alpine3.14

- [`0.13.0-python3.6-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/alpine3.14/runtime/Dockerfile)
- [`0.13.0-python3.7-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/alpine3.14/runtime/Dockerfile)
- [`0.13.0-python3.8-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/master/docker/generated/model-server/alpine3.14/runtime/Dockerfile)
