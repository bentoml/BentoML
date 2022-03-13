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


FROM bento-server:base-python$PYTHON_VERSION-debian11 as base


FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base as base-s390x

FROM base-${TARGETARCH}

COPY --from=xx / /

ARG TARGETARCH

ARG TARGETPLATFORM

LABEL maintainer "BentoML Team <contact@bentoml.com>"

ENV BENTOML_VERSION=1.0.0a6

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir