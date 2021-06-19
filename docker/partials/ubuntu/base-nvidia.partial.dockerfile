ARG CUDA=${CUDA}

FROM nvidia/cuda:${CUDA}.0-base-ubuntu${OS_VERSION}

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

ARG CUDA
ARG CUDNN_VERSION=${CUDNN_VERSION}
ARG CUDNN_MAJOR_VERSION=${CUDNN_MAJOR_VERSION}

RUN apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
        ca-certificates gnupg2 wget build-essential git \
        libcublas-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION}-1+cuda${CUDA} \
    && apt-get clean \
    && apt-mark hold libcudnn${CUDNN_MAJOR_VERSION} libcublas-${CUDA/./-} \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH