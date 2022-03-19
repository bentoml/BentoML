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

ENV BENTOML_VERSION=1.0.0a6

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir