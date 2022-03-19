# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM aarnphm/bento-server:base-python$PYTHON_VERSION-debian10 as stage_cached

FROM --platform=$BUILDPLATFORM aarnphm/bento-server:base-python$PYTHON_VERSION-debian10 as base

RUN --mount=type=cache,from=stage_cached,sharing=shared,target=/var/cache/apt \
    --mount=type=cache,from=stage_cached,sharing=shared,target=/var/lib/apt \
    xx-apt-get install -q -y --no-install-recommends git \
    && xx-apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
    pip install -U git+https://github.com/bentoml/bentoml.git@main

FROM base as base-amd64

FROM base as base-arm

FROM base as base-ppc64le

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM

RUN xx-apt-get purge -y -q curl git \
    && rm -rf /var/lib/{apt,dpkg,cache,log}