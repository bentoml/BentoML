# syntax = docker/dockerfile:1.4
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


ARG PYTHON_VERSION


FROM bento-server:base-python$PYTHON_VERSION-ubi7 as base


FROM base as base-amd64

FROM base as base-s390x

FROM base as base-ppc64le

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.com>"

WORKDIR /

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main