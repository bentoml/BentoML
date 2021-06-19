ARG CUDA_VERSION=${CUDA_VERSION}

FROM nvidia/cuda:${CUDA_VERSION}.0-base-centos${OS_VERSION}

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION

# define all cuda-related deps.
ARG CUDNN_VERSION=${CUDNN_VERSION}
ARG CUDNN_MAJOR_VERSION=${CUDNN_MAJOR_VERSION}
ARG CUBLAS_VERSION=${CUBLAS_VERSION}
ARG CURAND_VERSION=${CURAND_VERSION}
ARG CUSPARSE_VERSION=${CUSPARSE_VERSION}
ARG CUFFT_VERSION=${CUFFT_VERSION}
ARG CUSOLVER_VERSION=${CUSOLVER_VERSION}

RUN yum install -y \
        wget git gcc gcc-c++ make \
        libcublas-${CUDA_VERSION/./-}-${CUBLAS_VERSION}-1 \
        libcurand-${CUDA_VERSION/./-}-${CURAND_VERSION}-1 \
        libcusparse-${CUDA_VERSION/./-}-${CUSPARSE_VERSION}-1 \
        libcufft-${CUDA_VERSION/./-}-${CUFFT_VERSION}-1 \
        libcusolver-${CUDA_VERSION/./-}-${CUSOLVER_VERSION}-1 \
        libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION}-1.cuda${CUDA_VERSION} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV LD_LIBRARY_PATH /usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH