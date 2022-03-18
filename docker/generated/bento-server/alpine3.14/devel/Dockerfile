# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM aarnphm/bento-server:base-python$PYTHON_VERSION-alpine3.14 as base

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ENV BENTOML_VERSION=1.0.0a6

RUN xx-apk --no-cache --update add build-base git

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main

FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base as base-s390x

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM
