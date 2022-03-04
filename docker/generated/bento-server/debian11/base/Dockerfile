# syntax = docker/dockerfile:1.2
#
# ╔═════════════════════════════════════════════╗
# ║ THIS IS A GENERATED DOCKERFILE. DO NOT EDIT ║
# ╚═════════════════════════════════════════════╝
#


FROM debian:11-slim as base

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

FROM base as base-arm

FROM base as base-386

FROM base as base-s390x

FROM base as base-riscv64

FROM base-${TARGETARCH}

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

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates bash curl wget git gnupg build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    apt-get clean && apt-get autoremove -y && \
    apt-get purge -y wget

# Install python via conda
RUN /opt/conda/bin/conda install -y python=$PYTHON_VERSION pip && \
    /opt/conda/bin/conda clean -afy

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc