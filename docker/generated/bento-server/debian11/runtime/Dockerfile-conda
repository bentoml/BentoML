# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM bentoml/bento-server:base-python$PYTHON_VERSION-debian11 as base

ENV BENTOML_VERSION=1.0.0a6

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir

FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base as base-s390x

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM