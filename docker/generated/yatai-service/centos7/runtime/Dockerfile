# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM yatai-service:base-python$PYTHON_VERSION-centos7 as build_image

RUN pip install bentoml[yatai_service]==$BENTOML_VERSION --no-cache-dir && \
    yum autoremove -y curl && \
    yum clean all && \
    rm -rf /var/cache/yum/*

COPY tools/yatai-service/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "yatai-service-start"]
