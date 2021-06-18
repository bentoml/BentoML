# syntax=docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

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

RUN --mount=type=cache,target=/var/cache/yum \
        yum install -y \
            curl git gcc gcc-c++ make \
            libcublas-${CUDA_VERSION/./-}-${CUBLAS_VERSION}-1 \
            libcurand-${CUDA_VERSION/./-}-${CURAND_VERSION}-1 \
            libcusparse-${CUDA_VERSION/./-}-${CUSPARSE_VERSION}-1 \
            libcufft-${CUDA_VERSION/./-}-${CUFFT_VERSION}-1 \
            libcusolver-${CUDA_VERSION/./-}-${CUSOLVER_VERSION}-1 \
            libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION}-1.cuda${CUDA_VERSION} \
        && yum clean all \
        && rm -rf /var/cache/yum/*

ENV LD_LIBRARY_PATH /usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

# we will install python from conda
RUN curl -fSsL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -ya

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc

ARG UID=1034
ARG GID=1034
RUN groupadd -g $GID -o bentoml && useradd -m -u $UID -g $GID -o -r bentoml

USER bentoml
WORKDIR $HOME
RUN git clone https://github.com/bentoml/BentoML.git && \
    cd BentoML && \
    make install-local

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]