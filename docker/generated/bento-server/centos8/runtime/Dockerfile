# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-centos8 as build_image

ENV BENTOML_VERSION=1.0.0

RUN pip install bentoml[model_server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/bento-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]