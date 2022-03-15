# syntax = docker/dockerfile:1.4
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================



ARG PYTHON_VERSION

FROM aarnphm/bento-server:base-python$PYTHON_VERSION-alpine3.14 as base


FROM base as base-amd64

FROM base as base-arm64

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM

LABEL maintainer "BentoML Team <contact@bentoml.com>"

ENV BENTOML_VERSION=1.0.0a6

RUN conda create -n bentoml-python${PYTHON_VERSION} python=$PYTHON_VERSION anaconda && \
        conda activate bentoml-python${PYTHON_VERSION}

# bash is needed to run bentoml.
RUN conda install -y python=$PYTHON_VERSION pip \
    && conda clean -afy \
    && xx-apk add --no-cache bash git build-base

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main
