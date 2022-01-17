# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-debian as build_image

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" >/etc/apt/sources.list.d/cuda.list && \
    apt-get purge --auto-remove -y curl && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-11-3=11.3.109-1 \
        libcublas-11-3=11.5.1.109-1 \
        libcurand-11-3=10.2.4.109-1 \
        libcusparse-11-3=11.6.0.109-1 \
        libcufft-11-3=10.4.2.109-1 \
        libcusolver-11-3=11.1.2.109-1 \
        libcudnn8=8.2.0.53-1+cuda11.3 && \
    ln -s cuda-11.3 /usr/local/cuda && \
    apt-mark hold libcudnn8 libcublas-11-3=11.5.1.109-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# nvidia-container-runtime, which is needed for nvidia-docker
# https://github.com/NVIDIA/nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.3 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"

FROM build_image as release_image

ENV BENTOML_VERSION=1.0.0

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir && \
    rm -rf /var/lib/apt/lists/*