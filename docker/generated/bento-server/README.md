## Model Serving Made Easy  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20Machine%20Learning%20Model%20Serving%20Made%20Easy%20&url=https://github.com/bentoml/BentoML&via=bentomlai&hashtags=mlops,modelserving,ML,AI,machinelearning,bentoml)

[![pypi_status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![actions_status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)

BentoML let you create machine learning powered prediction service in minutes and bridges the gap between data science and DevOps.

👉 Join our [slack community](https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg)


## Why BentoML

- The easiest way to get your ML models into production.
- High performance model serving, all in Python.
- Package your model once and deploy it anywhere.
- Support all major ML model training [frameworks](https://docs.bentoml.org/en/latest/frameworks.html).

## Getting Started

- [Quickstart guide](https://docs.bentoml.org/en/latest/quickstart.html) will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.
- [Main concepts](https://docs.bentoml.org/en/latest/concepts.html) will give a comprehensive tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.
- Playground notebook gets your hands dirty in a notebook environment, for you to try out all the core features in BentoML.
- [ML Frameworks](https://docs.bentoml.org/en/latest/frameworks.html) lays out best practices and example usages by the ML framework used for training models.
- [Advanced Guides](https://docs.bentoml.org/en/latest/guides/index.html) show cases advanced features in BentoML, including GPU support, inference graph, monitoring, and customizing docker environment etc.


## BentoServer base images

There are three type of BentoServer docker base image:

| Image Type | Description                                | Supported OS                                          | Usage                             |
|------------|--------------------------------------------|-------------------------------------------------------|-----------------------------------|
| `runtime`  | contains latest BentoML releases from PyPI | `debian`, `centos{7,8}`, `amazonlinux2`, `alpine3.14` | production ready                  |
| `cudnn`    | runtime + support for CUDA-enabled GPU     | `debian`, `centos{7,8}`                               | production ready with GPU support |
| `devel`    | nightly build from development branch      | `debian`, `centos{7,8}`                               | for development use only          |

* Note: currently there's no nightly devel image with GPU support.

The final docker image tags will have the following format:

```markdown
<release_type>-<python_version>-<distros>-<suffix>
   │             │                │        │
   │             │                │        └─> additional suffix, differentiate runtime and cudnn releases
   │             │                └─> formatted <dist><dist_version>, e.g: ami2, debian, centos7
   │             └─> Supported Python version: python3.7 | python3.8 | python3.9
   └─>  Release type: devel or official BentoML release (e.g: 1.0.0)
```

Example image tags:
- `bento-server:devel-python3.7-debian`
- `bento-server:1.0.0-python3.8-centos8-cudnn`
- `bento-server:1.0.0-python3.7-ami2-runtime`


## Latest tags for `bento-server 1.0.0`
- [`1.0.0-python3.7-debian-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/cudnn/Dockerfile)
- [`1.0.0-python3.7-debian-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/runtime/Dockerfile)
- [`1.0.0-python3.8-debian-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/cudnn/Dockerfile)
- [`1.0.0-python3.8-debian-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/runtime/Dockerfile)
- [`1.0.0-python3.9-debian-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/cudnn/Dockerfile)
- [`1.0.0-python3.9-debian-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/runtime/Dockerfile)
- [`devel-python3.7-debian`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/devel/Dockerfile)
- [`devel-python3.8-debian`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/devel/Dockerfile)
- [`devel-python3.9-debian`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/devel/Dockerfile)

### Centos8

*WARNING*: POSSIBLE MISSING IMAGE TAGS

Centos upstream images often fail security scans, thus there might be some images missing. Please refers to [Issues section](https://github.com/bentoml/BentoML/issues) for security notices.

- [`1.0.0-python3.7-centos8-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/cudnn/Dockerfile)
- [`1.0.0-python3.7-centos8-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/runtime/Dockerfile)
- [`1.0.0-python3.8-centos8-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/cudnn/Dockerfile)
- [`1.0.0-python3.8-centos8-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/runtime/Dockerfile)
- [`1.0.0-python3.9-centos8-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/cudnn/Dockerfile)
- [`1.0.0-python3.9-centos8-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/runtime/Dockerfile)
- [`devel-python3.7-centos8`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/devel/Dockerfile)
- [`devel-python3.8-centos8`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/devel/Dockerfile)
- [`devel-python3.9-centos8`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos8/devel/Dockerfile)

### Centos7

*WARNING*: POSSIBLE MISSING IMAGE TAGS

Centos upstream images often fail security scans, thus there might be some images missing. Please refers to [Issues section](https://github.com/bentoml/BentoML/issues) for security notices.

- [`1.0.0-python3.7-centos7-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/cudnn/Dockerfile)
- [`1.0.0-python3.7-centos7-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/runtime/Dockerfile)
- [`1.0.0-python3.8-centos7-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/cudnn/Dockerfile)
- [`1.0.0-python3.8-centos7-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/runtime/Dockerfile)
- [`1.0.0-python3.9-centos7-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/cudnn/Dockerfile)
- [`1.0.0-python3.9-centos7-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/runtime/Dockerfile)
- [`devel-python3.7-centos7`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/devel/Dockerfile)
- [`devel-python3.8-centos7`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/devel/Dockerfile)
- [`devel-python3.9-centos7`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/centos7/devel/Dockerfile)

### Amazonlinux2

- [`1.0.0-python3.7-amazonlinux2-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/amazonlinux2/runtime/Dockerfile)
- [`1.0.0-python3.8-amazonlinux2-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/amazonlinux2/runtime/Dockerfile)
- [`1.0.0-python3.9-amazonlinux2-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/amazonlinux2/runtime/Dockerfile)

### Alpine3.14

- [`1.0.0-python3.7-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/alpine3.14/runtime/Dockerfile)
- [`1.0.0-python3.8-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/alpine3.14/runtime/Dockerfile)
- [`1.0.0-python3.9-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/alpine3.14/runtime/Dockerfile)
