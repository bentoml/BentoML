# syntax = docker/dockerfile:1.2
#
# ╔═════════════════════════════════════════════╗
# ║ THIS IS A GENERATED DOCKERFILE. DO NOT EDIT ║
# ╚═════════════════════════════════════════════╝
#


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-ubi8 as base

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.ai>"

WORKDIR /

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main