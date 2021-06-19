# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM ubuntu:${OS_VERSION}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
        ca-certificates gnupg2 wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG OS_VERSION

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

# we will install python from conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -ya

RUN pip install bentoml[model-server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]