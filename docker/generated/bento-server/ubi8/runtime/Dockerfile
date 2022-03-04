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

ENV BENTOML_VERSION=1.0.0a5

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir