# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM aarnphm/bento-server:base-python$PYTHON_VERSION-ubi8 as base

FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base-${TARGETARCH}

ARG TARGETARCH

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main

RUN yum remove -y gcc curl make git \
    && yum clean all \
    && rm -rf /var/cache/yum