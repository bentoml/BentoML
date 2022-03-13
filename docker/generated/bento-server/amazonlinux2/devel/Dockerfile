# syntax = docker/dockerfile:1.3
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


ARG XX_VERSION=1.1.0


FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-amazonlinux2 as base

COPY --from=xx / /

FROM base as base-arm64

FROM base as base-amd64

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.com>"

WORKDIR /

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main