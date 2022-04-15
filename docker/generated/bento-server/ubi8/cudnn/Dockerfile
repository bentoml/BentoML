# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


ARG PYTHON_VERSION


FROM bentoml/bento-server:base-python$PYTHON_VERSION-ubi8 as base

ENV NVIDIA_CUDA_LIB_VERSION 11.5.1-1

FROM base as base-amd64
ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=460,driver<461 brand=tesla,driver>=470,driver<471"
ENV NVIDIA_CUDA_CUDART_VERSION 11.5.117-1

COPY cuda.repo-amd64 /etc/yum.repos.d/cuda.repo
COPY nvidia-ml.repo-amd64 /etc/yum.repos.d/nvidia-ml.repo

ENV NVIDIA_NVTX_VERSION 11.5.114-1

ENV NVIDIA_LIBNPP_VERSION 11.5.1.107-1
ENV NVIDIA_LIBNPP_PACKAGE libnpp-11-5-${NVIDIA_LIBNPP_VERSION}

ENV NVIDIA_LIBCUBLAS_VERSION 11.7.4.6-1

ENV NVIDIA_LIBNCCL_PACKAGE_NAME libnccl
ENV NVIDIA_LIBNCCL_PACKAGE_VERSION 2.11.4-1
ENV NVIDIA_LIBNCCL_VERSION 2.11.4
ENV NCCL_VERSION 2.11.4
ENV NVIDIA_LIBNCCL_PACKAGE ${NVIDIA_LIBNCCL_PACKAGE_NAME}-${NVIDIA_LIBNCCL_PACKAGE_VERSION}+cuda11.5
ENV NVIDIA_CUDNN_VERSION 8.3.1.22-1
ENV NVIDIA_CUDNN_PACKAGE libcudnn8-${NVIDIA_CUDNN_VERSION}.cuda11.5


FROM base as base-arm64
ENV NVARCH sbsa

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5"
ENV NVIDIA_CUDA_CUDART_VERSION 11.5.117-1

COPY cuda.repo-arm64v8 /etc/yum.repos.d/cuda.repo
COPY nvidia-ml.repo-arm64v8 /etc/yum.repos.d/nvidia-ml.repo

ENV NVIDIA_NVTX_VERSION 11.5.114-1

ENV NVIDIA_LIBNPP_VERSION 11.5.1.107-1
ENV NVIDIA_LIBNPP_PACKAGE libnpp-11-5-${NVIDIA_LIBNPP_VERSION}

ENV NVIDIA_LIBCUBLAS_VERSION 11.7.4.6-1

ENV NVIDIA_LIBNCCL_PACKAGE_NAME libnccl
ENV NVIDIA_LIBNCCL_PACKAGE_VERSION 2.11.4-1
ENV NVIDIA_LIBNCCL_VERSION 2.11.4
ENV NCCL_VERSION 2.11.4
ENV NVIDIA_LIBNCCL_PACKAGE ${NVIDIA_LIBNCCL_PACKAGE_NAME}-${NVIDIA_LIBNCCL_PACKAGE_VERSION}+cuda11.5
ENV NVIDIA_CUDNN_VERSION 8.3.1.22-1
ENV NVIDIA_CUDNN_PACKAGE libcudnn8-${NVIDIA_CUDNN_VERSION}.cuda11.5


FROM base as base-ppc64le
ENV NVARCH ppc64le

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5"
ENV NVIDIA_CUDA_CUDART_VERSION 11.5.117-1

COPY cuda.repo-ppc64le /etc/yum.repos.d/cuda.repo
COPY nvidia-ml.repo-ppc64le /etc/yum.repos.d/nvidia-ml.repo

ENV NVIDIA_NVTX_VERSION 11.5.114-1

ENV NVIDIA_LIBNPP_VERSION 11.5.1.107-1
ENV NVIDIA_LIBNPP_PACKAGE libnpp-11-5-${NVIDIA_LIBNPP_VERSION}

ENV NVIDIA_LIBCUBLAS_VERSION 11.7.4.6-1

ENV NVIDIA_LIBNCCL_PACKAGE_NAME libnccl
ENV NVIDIA_LIBNCCL_PACKAGE_VERSION 2.11.4-1
ENV NVIDIA_LIBNCCL_VERSION 2.11.4
ENV NCCL_VERSION 2.11.4
ENV NVIDIA_LIBNCCL_PACKAGE ${NVIDIA_LIBNCCL_PACKAGE_NAME}-${NVIDIA_LIBNCCL_PACKAGE_VERSION}+cuda11.5

ENV NVIDIA_CUDNN_VERSION 8.3.1.22-1
ENV NVIDIA_CUDNN_PACKAGE libcudnn8-${NVIDIA_CUDNN_VERSION}.cuda11.5

FROM base-${TARGETARCH}

ARG TARGETARCH

ENV CUDA_VERSION 11.5.1

USER root

# nvidia-docker 1.0
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${NVARCH}/7fa2af80.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA && \
    echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN --mount=type=cache,sharing=shared,target=/var/cache/yum \
    yum upgrade -y && yum install -y \
    cuda-cudart-11-5-${NVIDIA_CUDA_CUDART_VERSION} \
    cuda-compat-11-5 \
    && ln -s cuda-11.5 /usr/local/cuda

RUN --mount=type=cache,sharing=shared,target=/var/cache/yum \
    yum install -y \
    cuda-libraries-11-5-${NVIDIA_CUDA_LIB_VERSION} \
    cuda-nvtx-11-5-${NVIDIA_NVTX_VERSION} \
    ${NVIDIA_LIBNPP_PACKAGE} \
    libcublas-11-5-${NVIDIA_LIBCUBLAS_VERSION} \
    ${NVIDIA_LIBNCCL_PACKAGE} \
    ${NVIDIA_CUDNN_PACKAGE}

ENV BENTOML_VERSION=1.0.0a7

RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
    pip install bentoml==${BENTOML_VERSION} --no-cache-dir

RUN yum clean all && rm -rf /var/cache/yum