# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


ARG PYTHON_VERSION

FROM aarnphm/bento-server:base-python$PYTHON_VERSION-amazonlinux2 as base

FROM base as base-arm64

FROM base as base-amd64

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.com>"

WORKDIR /

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main