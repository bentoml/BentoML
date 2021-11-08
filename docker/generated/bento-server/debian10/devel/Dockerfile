# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM bento-server:base-python$PYTHON_VERSION-slim as build_image

WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git && \
    cd BentoML && \
    pip install --editable . && \
    apt-get purge --auto-remove -y curl && \
    rm -rf /var/lib/apt/lists/*