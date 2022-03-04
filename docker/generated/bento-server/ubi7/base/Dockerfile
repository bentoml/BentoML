# syntax = docker/dockerfile:1.3
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


FROM registry.access.redhat.com/ubi7/ubi:latest as base

FROM base as base-amd64
ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=460,driver<461 brand=tesla,driver>=470,driver<471"
ENV NV_CUDA_CUDART_VERSION 11.5.117-1


FROM base as base-arm64
ENV NVARCH sbsa

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5"
ENV NV_CUDA_CUDART_VERSION 11.5.117-1


FROM base as base-ppc64le
ENV NVARCH ppc64le

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5"
ENV NV_CUDA_CUDART_VERSION 11.5.117-1



FROM base as base-s390x

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "BentoML Team <contact@bentoml.ai>"

# setup ENV and ARG
ARG PYTHON_VERSION
ENV PYTHONDONTWRITEBYTECODE=1

ENV PATH /opt/conda/bin:$PATH

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN yum upgrade -y \
    && yum install -y wget git ca-certificates curl gcc gcc-c++ make \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname -s)-$(uname -m).sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    yum clean all && \
    yum autoremove -y wget && \
    rm -rf /var/cache/yum

RUN /opt/conda/bin/conda install -y python=$PYTHON_VERSION pip && \
    /opt/conda/bin/conda clean -afy

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc