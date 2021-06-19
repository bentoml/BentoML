# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

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

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

# we will install python from conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -ya

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc

WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git \
    && cd BentoML \
    && pip install --editable .

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]