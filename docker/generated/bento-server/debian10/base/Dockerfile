# syntax = docker/dockerfile:1.3
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================



ARG XX_VERSION=1.1.0


FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx


FROM debian:10-slim as base

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

FROM base as base-s390x

FROM base-${TARGETARCH}

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.ai>"

# setup ENV and ARG
ARG PYTHON_VERSION
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1

ENV PATH /opt/conda/bin:$PATH

ENV DEBIAN_FRONTEND noninteractive

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN xx-apt-get update && \
    xx-apt-get install -q -y --no-install-recommends \
    ca-certificates bash curl wget git gnupg build-essential && \
    xx-apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=latest

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
        SHA256SUM="1faed9abecf4a4ddd4e0d8891fc2cdaa3394c51e877af14ad6b9d4aadb4e90d8"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
        SHA256SUM="4879820a10718743f945d88ef142c3a4b30dfc8e448d1ca08e019586374b773f"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh"; \
        SHA256SUM="fa92ee4773611f58ed9333f977d32bbb64769292f605d518732183be1f3321fa"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    echo "${SHA256SUM} miniconda.sh" > shasum && \
    if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Install python via conda
RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip && \
    /opt/conda/bin/conda clean -afy

COPY hack/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc