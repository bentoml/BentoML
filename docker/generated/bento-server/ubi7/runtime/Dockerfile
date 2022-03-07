# syntax = docker/dockerfile:1.3
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

ENV BENTOML_VERSION=1.0.0-a5

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir