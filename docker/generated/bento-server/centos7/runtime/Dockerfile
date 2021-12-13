# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-centos7 as build_image

ENV BENTOML_VERSION=1.0.0

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir