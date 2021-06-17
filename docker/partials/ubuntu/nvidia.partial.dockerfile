ARG CUDA=${CUDA_VERSION}
ARG CUDNN_MAJOR_VER=${CUDNN_MAJOR_VER}

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates gnupg2 curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
FROM nvidia/cuda:${CUDA}-base-${OS}
# needed for string substitutions
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libcublas-${CUDA/./-} \
    libcurand-${CUDA/./-} \
    libcusparse-${CUDA/./-} \
    libcufft-${CUDA/./-} \
    libcusolver-${CUDA/./-} \
    libcudnn${CUDNN_MAJOR_VER}=${CUDNN}-1+cuda${CUDA} \
    && apt-get clean \
    && apt-mark hold libcudnn${CUDNN_MAJOR_VER} libcublas-${CUDA/./-} \
    && rm -rf /var/lib/apt/lists/*