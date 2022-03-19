# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

ARG XX_VERSION=1.1.0

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

FROM python:${PYTHON_VERSION}-slim-buster as stage_cached

FROM --platform=$BUILDPLATFORM python:${PYTHON_VERSION}-slim-buster as base

FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base-${TARGETARCH}

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,from=stage_cached,sharing=shared,target=/var/cache/apt \
    --mount=type=cache,from=stage_cached,sharing=shared,target=/var/lib/apt \
    xx-apt-get install -q -y --no-install-recommends bash gcc python3-dev \
    && xx-apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ARG PYTHON_VERSION

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8

ENV PYTHONDONTWRITEBYTECODE=1

ENV DEBIAN_FRONTEND noninteractive

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]