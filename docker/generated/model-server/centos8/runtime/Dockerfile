# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM model-server:base-python$PYTHON_VERSION-centos8 as build_image

RUN pip install bentoml[model_server]==$BENTOML_VERSION --no-cache-dir && \
    yum autoremove -y curl && \
    yum clean all && \
    rm -rf /var/cache/yum/*

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]
