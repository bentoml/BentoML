# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG UBIFORMAT

FROM registry.access.redhat.com/ubi8/$UBIFORMAT:latest as base

FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG PYTHON_VERSION

ENV PYTHONDONTWRITEBYTECODE=1

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

USER root

RUN --mount=type=cache,sharing=shared,target=/var/cache/yum \
    yum upgrade -y \
    && yum install -y wget git ca-certificates curl gcc gcc-c++ make \
    && yum clean all