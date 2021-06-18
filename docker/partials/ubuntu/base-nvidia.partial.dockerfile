ARG CUDA_VERSION=${CUDA_VERSION}

FROM nvidia/cuda:${CUDA_VERSION}.0-base-ubuntu${OS_VERSION}

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION
ARG CUDNN_VERSION=${CUDNN_VERSION}
ARG CUDNN_MAJOR_VERSION=${CUDNN_MAJOR_VERSION}

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
            ca-certificates gnupg2 curl build-essential git \
            libcublas-${CUDA_VERSION/./-} \
            libcurand-${CUDA_VERSION/./-} \
            libcusparse-${CUDA_VERSION/./-} \
            libcufft-${CUDA_VERSION/./-} \
            libcusolver-${CUDA_VERSION/./-} \
            libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION}-1+cuda${CUDA_VERSION} \
    && apt-get clean \
    && apt-mark hold libcudnn${CUDNN_MAJOR_VERSION} libcublas-${CUDA_VERSION/./-} \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH