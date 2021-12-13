# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-alpine3.14 as build_image

ENV BENTOML_VERSION=1.0.0

# bash is needed to run bentoml.
RUN conda install -y python=$PYTHON_VERSION pip \
    && conda clean -afy \
    && apk add --no-cache bash git build-base

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir