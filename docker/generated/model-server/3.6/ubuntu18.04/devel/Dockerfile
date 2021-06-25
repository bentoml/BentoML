# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM model-server:python3.6-ubuntu18.04-base

WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git && \
    cd BentoML && \
    pip install --editable . && \
    apt-get purge --autoremove -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]