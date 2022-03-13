# syntax = docker/dockerfile:1.3
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================



ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-alpine3.14 as base


FROM base as base-amd64

FROM base as base-arm64

FROM base as base-s390x

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM

LABEL maintainer "BentoML Team <contact@bentoml.com>"

ENV BENTOML_VERSION=1.0.0a6

# bash is needed to run bentoml.
RUN conda install -y python=$PYTHON_VERSION pip \
    && conda clean -afy \
    && xx-apk add --no-cache bash git build-base

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir
