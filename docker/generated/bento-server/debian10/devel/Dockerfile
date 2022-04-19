# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM aarnphm/bento-server:base-python$PYTHON_VERSION-debian10 as stage_cached

FROM aarnphm/bento-server:base-python$PYTHON_VERSION-debian10 as base

RUN xx-apt-get install -q -y --no-install-recommends git \
    && xx-apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
    pip install -U git+https://github.com/bentoml/bentoml.git@main

FROM base as base-amd64

FROM base as base-arm64

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM

RUN rm -rf /var/lib/{apt,dpkg,cache,log}