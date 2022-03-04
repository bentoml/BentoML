# syntax = docker/dockerfile:1.2
#
# ╔═════════════════════════════════════════════╗
# ║ THIS IS A GENERATED DOCKERFILE. DO NOT EDIT ║
# ╚═════════════════════════════════════════════╝
#


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-debian11 as base

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.ai>"

ENV BENTOML_VERSION=1.0.0a5

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir