# Unified Model Serving Framework  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=BentoML:%20The%20Unified%20Model%20Serving%20Framework%20&url=https://github.com/bentoml&via=bentomlai&hashtags=mlops,bentoml)

BentoML is an open platform that simplifies ML model deployment and enables you to serve your models at production scale in minutes

👉 [Pop into our Slack community!](https://join.slack.bentoml.org) We're happy to help with any issue you face or even just to meet you and hear what you're working on :)

[![pypi_status](https://img.shields.io/pypi/v/bentoml.svg)](https://pypi.org/project/BentoML)
[![downloads](https://pepy.tech/badge/bentoml)](https://pepy.tech/project/bentoml)
[![actions_status](https://github.com/bentoml/bentoml/workflows/BentoML-CI/badge.svg)](https://github.com/bentoml/bentoml/actions)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.org/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://join.slack.bentoml.org)


## Why BentoML ##

- The easiest way to turn your ML models into production-ready API endpoints.
- High performance model serving, all in Python.
- Standardized model packaging and ML service definition to streamline deployment.
- Support all major machine-learning training [frameworks](https://docs.bentoml.org/en/latest/frameworks/index.html).
- Deploy and operate ML serving workload at scale on Kubernetes via [Yatai](https://github.com/bentoml/yatai).

## Getting Started ##

- [Quickstart guide](https://docs.bentoml.org/en/latest/quickstart.html) will show you a simple example of using BentoML in action. In under 10 minutes, you'll be able to serve your ML model over an HTTP API endpoint, and build a docker image that is ready to be deployed in production.
- [Main concepts](https://docs.bentoml.org/en/latest/concepts/index.html) will give a comprehensive tour of BentoML's components and introduce you to its philosophy. After reading, you will see what drives BentoML's design, and know what `bento` and `runner` stands for.
- [ML Frameworks](https://docs.bentoml.org/en/latest/frameworks/index.html) lays out best practices and example usages by the ML framework used for training models.
- [Advanced Guides](https://docs.bentoml.org/en/latest/guides/index.html) showcases advanced features in BentoML, including GPU support, inference graph, monitoring, and customizing docker environment etc.
- Check out other projects from the [BentoML team](https://github.com/bentoml):
  - [🦄️ Yatai](https://github.com/bentoml/yatai): Run BentoML workflow at scale on Kubernetes
  - [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment with BentoML on cloud platforms


## BentoServer base images

There are three type of BentoServer docker base image:

| Image Type | Description                                | Supported OS                                          | Usage                             |
|------------|--------------------------------------------|-------------------------------------------------------|-----------------------------------|
| `runtime`  | contains latest BentoML releases from PyPI | `debian`, `ubi{7,8}`, `amazonlinux2`, `alpine3.14`    | production ready                  |
| `cudnn`    | runtime + support for CUDA-enabled GPU     | `debian`, `ubi{7,8}`                                  | production ready with GPU support |
| `devel`    | nightly build from development branch      | `debian`, `ubi{7,8}`                                  | for development use only          |

* Note: currently there's no nightly devel image with GPU support.

The final docker image tags will have the following format:

```markdown
<release_type>-<python_version>-<distros>-<suffix>
   │             │                │        │
   │             │                │        └─> additional suffix, differentiate runtime and cudnn releases
   │             │                └─> formatted <dist><dist_version>, e.g: ami2, debian, ubi7
   │             └─> Supported Python version: python3.7 | python3.8 | python3.9
   └─>  Release type: devel or official BentoML release (e.g: 1.0.0)
```

Example image tags:
- `bento-server:devel-python3.7-debian`
- `bento-server:1.0.0-python3.8-ubi8-cudnn`
- `bento-server:1.0.0-python3.7-ami2-runtime`

## Latest tags for `bento-server 1.0.0a5`


### debian 11 [_arm32v5_, _arm64v8_, _arm32v7_, _i386_, _riscv64_, _s390x_, _amd64_, _arm32v6_, _ppc64le_]

- [`1.0.0a5-python3.7-debian11-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/cudnn/Dockerfile)
- [`1.0.0a5-python3.7-debian11-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/runtime/Dockerfile)
- [`1.0.0a5-python3.8-debian11-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/cudnn/Dockerfile)
- [`1.0.0a5-python3.8-debian11-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/runtime/Dockerfile)
- [`1.0.0a5-python3.9-debian11-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/cudnn/Dockerfile)
- [`1.0.0a5-python3.9-debian11-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/runtime/Dockerfile)
- [`devel-python3.7-debian11`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/devel/Dockerfile)
- [`devel-python3.8-debian11`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/devel/Dockerfile)
- [`devel-python3.9-debian11`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian11/devel/Dockerfile)

### debian 10 [_arm32v5_, _arm64v8_, _arm32v7_, _i386_, _riscv64_, _s390x_, _amd64_, _arm32v6_, _ppc64le_]

- [`1.0.0a5-python3.7-debian10-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/cudnn/Dockerfile)
- [`1.0.0a5-python3.7-debian10-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/runtime/Dockerfile)
- [`1.0.0a5-python3.8-debian10-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/cudnn/Dockerfile)
- [`1.0.0a5-python3.8-debian10-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/runtime/Dockerfile)
- [`1.0.0a5-python3.9-debian10-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/cudnn/Dockerfile)
- [`1.0.0a5-python3.9-debian10-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/runtime/Dockerfile)
- [`devel-python3.7-debian10`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/devel/Dockerfile)
- [`devel-python3.8-debian10`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/devel/Dockerfile)
- [`devel-python3.9-debian10`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/debian10/devel/Dockerfile)

### UBI 8 [_arm32v5_, _arm64v8_, _arm32v7_, _i386_, _riscv64_, _s390x_, _amd64_, _arm32v6_, _ppc64le_]

- [`1.0.0a5-python3.7-ubi8-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/cudnn/Dockerfile)
- [`1.0.0a5-python3.7-ubi8-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/runtime/Dockerfile)
- [`1.0.0a5-python3.8-ubi8-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/cudnn/Dockerfile)
- [`1.0.0a5-python3.8-ubi8-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/runtime/Dockerfile)
- [`1.0.0a5-python3.9-ubi8-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/cudnn/Dockerfile)
- [`1.0.0a5-python3.9-ubi8-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/runtime/Dockerfile)
- [`devel-python3.7-ubi8`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/devel/Dockerfile)
- [`devel-python3.8-ubi8`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/devel/Dockerfile)
- [`devel-python3.9-ubi8`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi8/devel/Dockerfile)

### UBI 7 [_arm32v5_, _arm64v8_, _arm32v7_, _i386_, _riscv64_, _s390x_, _amd64_, _arm32v6_, _ppc64le_]

- [`1.0.0a5-python3.7-ubi7-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/cudnn/Dockerfile)
- [`1.0.0a5-python3.7-ubi7-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/runtime/Dockerfile)
- [`1.0.0a5-python3.8-ubi7-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/cudnn/Dockerfile)
- [`1.0.0a5-python3.8-ubi7-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/runtime/Dockerfile)
- [`1.0.0a5-python3.9-ubi7-cudnn`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/cudnn/Dockerfile)
- [`1.0.0a5-python3.9-ubi7-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/runtime/Dockerfile)
- [`devel-python3.7-ubi7`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/devel/Dockerfile)
- [`devel-python3.8-ubi7`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/devel/Dockerfile)
- [`devel-python3.9-ubi7`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/ubi7/devel/Dockerfile)

### amazonlinux 2 [_arm32v5_, _arm64v8_, _arm32v7_, _i386_, _riscv64_, _s390x_, _amd64_, _arm32v6_, _ppc64le_]

- [`1.0.0a5-python3.7-amazonlinux2-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/amazonlinux2/runtime/Dockerfile)
- [`1.0.0a5-python3.8-amazonlinux2-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/amazonlinux2/runtime/Dockerfile)
- [`1.0.0a5-python3.9-amazonlinux2-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/amazonlinux2/runtime/Dockerfile)

### alpine 3.14 [_arm32v5_, _arm64v8_, _arm32v7_, _i386_, _riscv64_, _s390x_, _amd64_, _arm32v6_, _ppc64le_]

- [`1.0.0a5-python3.7-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/alpine3.14/runtime/Dockerfile)
- [`1.0.0a5-python3.8-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/alpine3.14/runtime/Dockerfile)
- [`1.0.0a5-python3.9-alpine3.14-runtime`](https://github.com/bentoml/BentoML/tree/main/docker/generated/bento-server/alpine3.14/runtime/Dockerfile)
