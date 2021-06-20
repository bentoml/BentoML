# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

FROM centos:${OS_VERSION} as base-image

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN yum upgrade -y \
    && yum install -y wget git gcc gcc-c++ ca-certificates make \
    && yum clean all \
    && rm -rf /var/cache/yum/*

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