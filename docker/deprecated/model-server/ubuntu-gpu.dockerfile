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

FROM ubuntu:${OS_VERSION} as build-image

# Redefine BENTOML_VERSION because of multistage
ARG BENTOML_VERSION

COPY --from=base-image /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH

RUN pip install bentoml[model_server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]