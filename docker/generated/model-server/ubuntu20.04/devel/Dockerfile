# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM model-server:python${PYTHON_VERSION}-ubuntu20.04-base as build_image

WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git && \
    cd BentoML && \
    pip install --editable . && \
    apt-get purge --autoremove -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]