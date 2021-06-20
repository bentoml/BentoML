# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

ARG CUDA=${CUDA}

FROM nvidia/cuda:${CUDA}.0-base-centos${OS_VERSION}

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

ARG CUDA

# define all cuda-related deps.
ARG CUDNN_VERSION=${CUDNN_VERSION}
ARG CUDNN_MAJOR_VERSION=${CUDNN_MAJOR_VERSION}
ARG CUBLAS_VERSION=${CUBLAS_VERSION}
ARG CURAND_VERSION=${CURAND_VERSION}
ARG CUSPARSE_VERSION=${CUSPARSE_VERSION}
ARG CUFFT_VERSION=${CUFFT_VERSION}
ARG CUSOLVER_VERSION=${CUSOLVER_VERSION}

RUN yum install -y \
        wget git gcc gcc-c++ make ca-certificates \
        libcublas-${CUDA/./-}-${CUBLAS_VERSION}-1 \
        libcurand-${CUDA/./-}-${CURAND_VERSION}-1 \
        libcusparse-${CUDA/./-}-${CUSPARSE_VERSION}-1 \
        libcufft-${CUDA/./-}-${CUFFT_VERSION}-1 \
        libcusolver-${CUDA/./-}-${CUSOLVER_VERSION}-1 \
        libcudnn${CUDNN_MAJOR_VERSION}-${CUDNN_VERSION}-1.cuda${CUDA} \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV LD_LIBRARY_PATH /usr/include/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

ENV PYTHONDONTWRITEBYTECODE=1

# we will install python from conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -afy

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc

WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git \
    && cd BentoML \
    && pip install --editable .

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]