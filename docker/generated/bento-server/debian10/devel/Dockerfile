# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


ARG PYTHON_VERSION


FROM bentoml/bento-server:base-python$PYTHON_VERSION-debian10 as base


FROM base as base-amd64

FROM base as base-arm

FROM base as base-ppc64le

FROM base as base-s390x

FROM base-${TARGETARCH}

ARG TARGETARCH

ARG TARGETPLATFORM

LABEL maintainer "BentoML Team <contact@bentoml.com>"

WORKDIR /

RUN pip install -U git+https://github.com/bentoml/bentoml.git@main && \
    xx-apt-get purge --auto-remove -y curl && \
    rm -rf /var/lib/apt/lists/*