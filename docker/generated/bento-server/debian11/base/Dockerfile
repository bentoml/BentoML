# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG XX_VERSION=1.1.0

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM python:${PYTHON_VERSION}-slim-bullseye as base

FROM base as base-amd64
ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=460,driver<461 brand=tesla,driver>=470,driver<471"

ENV NVIDIA_CUDA_CUDART_VERSION 11.5.117-1
ENV NVIDIA_CUDA_COMPAT_PACKAGE cuda-compat-11-5

ENV NVIDIA_ML_REPO_ENABLED 1
ENV NVIDIA_ML_REPO_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/${NVARCH}

FROM base as base-arm64
ENV NVARCH sbsa

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5"

ENV NVIDIA_CUDA_CUDART_VERSION 11.5.117-1

ENV NVIDIA_ML_REPO_ENABLED 1
ENV NVIDIA_ML_REPO_URL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/${NVARCH}

FROM base as base-ppc64le

FROM base-${TARGETARCH}

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

ARG PYTHON_VERSION

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8

ENV PYTHONDONTWRITEBYTECODE=1

ENV DEBIAN_FRONTEND noninteractive

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN xx-apt-get update \
    && xx-apt-get install -q -y --no-install-recommends gcc python3-dev \
    && xx-apt-get clean \
    && rm -rf /var/lib/{apt,dpkg,cache,log}