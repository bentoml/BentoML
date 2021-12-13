# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-centos8 as build_image

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/7fa2af80.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

COPY generated/bento-server/centos8/cudnn/cuda.repo /etc/yum.repos.d/cuda.repo
COPY generated/bento-server/centos8/cudnn/nvidia-ml.repo /etc/yum.repos.d/nvidia-ml.repo

RUN yum upgrade -y && yum install -y \
        cuda-cudart-11-3 \
        cuda-compat-11-3 \
        libcublas-11-3-11.5.1.109-1 \
        libcurand-11-3-10.2.4.109-1 \
        libcusparse-11-3-11.6.0.109-1 \
        libcufft-11-3-10.4.2.109-1 \
        libcusolver-11-3-11.1.2.109-1 \
        libcudnn8-8.2.0.53-1.cuda11.3 && \
    ln -s cuda-11.3 /usr/local/cuda && \
    yum clean all && \
    rm -rf /var/cache/yum

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# nvidia-container-runtime, which is needed for nvidia-docker
# https://github.com/NVIDIA/nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.3 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"

FROM build_image as release_image

ENV BENTOML_VERSION=1.0.0

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir