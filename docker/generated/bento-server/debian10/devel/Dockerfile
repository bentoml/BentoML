# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-debian as build_image

WORKDIR /
RUN pip install -U git+https://github.com/bentoml/bentoml.git@main && \
    apt-get purge --auto-remove -y curl && \
    rm -rf /var/lib/apt/lists/*