# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

FROM --platform=$BUILDPLATFORM aarnphm/bento-server:base-python$PYTHON_VERSION-debian11 as baseline

ARG TARGETPLATFORM

ARG TARGETARCH

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    xx-apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    if [ ! -z ${NVIDIA_ML_REPO_ENABLED} ]; then echo "deb ${NVIDIA_ML_REPO_URL} /" > /etc/apt/sources.list.d/nvidia-ml.list; fi

ENV CUDA_VERSION 11.5.1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    xx-apt-get install -y --no-install-recommends \
    cuda-cudart-11-5=${NVIDIA_CUDA_CUDART_VERSION} \
    ${NVIDIA_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.5 /usr/local/cuda

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

FROM baseline as base

ENV NVIDIA_CUDA_LIB_VERSION 11.5.1-1

FROM base as base-amd64

ENV NVIDIA_NVTX_VERSION 11.5.114-1
ENV NVIDIA_LIBNPP_VERSION 11.5.1.107-1
ENV NVIDIA_LIBNPP_PACKAGE libnpp-11-5=${NVIDIA_LIBNPP_VERSION}

ENV NVIDIA_LIBCUSPARSE_VERSION 11.7.0.107-1

ENV NVIDIA_LIBCUBLAS_PACKAGE_NAME libcublas-11-5
ENV NVIDIA_LIBCUBLAS_VERSION 11.7.4.6-1
ENV NVIDIA_LIBCUBLAS_PACKAGE ${NVIDIA_LIBCUBLAS_PACKAGE_NAME}=${NVIDIA_LIBCUBLAS_VERSION}

ENV NVIDIA_LIBNCCL_PACKAGE_NAME libnccl2
ENV NVIDIA_LIBNCCL_PACKAGE_VERSION 2.11.4-1
ENV NCCL_VERSION 2.11.4-1
ENV NVIDIA_LIBNCCL_PACKAGE ${NVIDIA_LIBNCCL_PACKAGE_NAME}=${NVIDIA_LIBNCCL_PACKAGE_VERSION}+cuda11.5
ENV NVIDIA_CUDNN_VERSION 8.3.1.22

ENV NVIDIA_CUDNN_PACKAGE "libcudnn8=$NVIDIA_CUDNN_VERSION-1+cuda11.5"
ENV NVIDIA_CUDNN_PACKAGE_NAME "libcudnn8"


FROM base as base-arm64

ENV NVIDIA_NVTX_VERSION 11.5.114-1
ENV NVIDIA_LIBNPP_VERSION 11.5.1.107-1
ENV NVIDIA_LIBNPP_PACKAGE libnpp-11-5=${NVIDIA_LIBNPP_VERSION}

ENV NVIDIA_LIBCUSPARSE_VERSION 11.7.0.107-1

ENV NVIDIA_LIBCUBLAS_PACKAGE_NAME libcublas-11-5
ENV NVIDIA_LIBCUBLAS_VERSION 11.7.4.6-1
ENV NVIDIA_LIBCUBLAS_PACKAGE ${NVIDIA_LIBCUBLAS_PACKAGE_NAME}=${NVIDIA_LIBCUBLAS_VERSION}

ENV NVIDIA_LIBNCCL_PACKAGE_NAME libnccl2
ENV NVIDIA_LIBNCCL_PACKAGE_VERSION 2.11.4-1
ENV NCCL_VERSION 2.11.4-1
ENV NVIDIA_LIBNCCL_PACKAGE ${NVIDIA_LIBNCCL_PACKAGE_NAME}=${NVIDIA_LIBNCCL_PACKAGE_VERSION}+cuda11.5

ENV NVIDIA_CUDNN_VERSION 8.3.1.22

ENV NVIDIA_CUDNN_PACKAGE "libcudnn8=$NVIDIA_CUDNN_VERSION-1+cuda11.5"
ENV NVIDIA_CUDNN_PACKAGE_NAME "libcudnn8"


FROM base as base-ppc64le

FROM base-${TARGETARCH}

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    xx-apt-get install -y --no-install-recommends \
    cuda-libraries-11-5=${NVIDIA_CUDA_LIB_VERSION} \
    ${NVIDIA_LIBNPP_PACKAGE} \
    cuda-nvtx-11-5=${NVIDIA_NVTX_VERSION} \
    libcusparse-11-5=${NVIDIA_LIBCUSPARSE_VERSION} \
    ${NVIDIA_LIBCUBLAS_PACKAGE} \
    ${NVIDIA_CUDNN_PACKAGE} \
    ${NVIDIA_LIBNCCL_PACKAGE}

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NVIDIA_CUDNN_PACKAGE_NAME} ${NVIDIA_LIBCUBLAS_PACKAGE_NAME} ${NVIDIA_LIBNCCL_PACKAGE_NAME}
ENV BENTOML_VERSION=1.0.0a6

RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
    pip install bentoml==${BENTOML_VERSION} --no-cache-dir