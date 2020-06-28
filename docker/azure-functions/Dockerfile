# Using specific version of docker image, due to issues at upstream. Once
# https://github.com/Azure/azure-functions-docker/issues/281 is resolved, we can update
# this.
ARG PYTHON_VERSION
FROM mcr.microsoft.com/azure-functions/python:3.0.13901-python$PYTHON_VERSION

# Install miniconda3
# https://hub.docker.com/r/continuumio/miniconda3/dockerfile
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
# Finish install miniconda3

# Install BentoML related
RUN apt-get update --fix-missing && \
    apt-get install -y build-essential && \
    apt-get clean

ARG BENTOML_VERSION
ENV BENTOML_VERSION=$BENTOML_VERSION

# pre-install BentoML base dependencies
RUN pip install bentoml==$BENTOML_VERSION
