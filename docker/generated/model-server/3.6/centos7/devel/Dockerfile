# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM model-server:python3.6-centos7-base

WORKDIR /
RUN git clone https://github.com/bentoml/BentoML.git && \
    cd BentoML && \
    pip install --editable . && \
    yum autoremove -y curl && \
    yum clean all && \
    rm -rf /var/cache/yum/*

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]