# syntax = docker/dockerfile:1.4
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================



ARG PYTHON_VERSION

FROM bentoml/bento-server:base-python$PYTHON_VERSION-ubi8 as base


FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base as base-s390x

FROM base-${TARGETARCH}

ARG TARGETARCH

ENV BENTOML_VERSION=1.0.0a6

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir