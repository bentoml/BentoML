# syntax = docker/dockerfile:1.3
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
# 
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-amazonlinux2 as base

FROM base-${TARGETARCH}

ARG TARGETARCH

ENV BENTOML_VERSION=1.0.0a5

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir