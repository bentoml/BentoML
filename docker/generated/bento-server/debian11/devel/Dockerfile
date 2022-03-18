# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM aarnphm/bento-server:base-python$PYTHON_VERSION-debian11 as base

RUN xx-apt-get install -q -y --no-install-recommends git

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main

RUN xx-apt-get purge --auto-remove -y git gcc python3-dev \
    && xx-apt-get clean \
    && rm -rf /var/lib/{apt,dpkg,cache,log}


FROM base as base-amd64

FROM base as base-arm

FROM base as base-ppc64le

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM
