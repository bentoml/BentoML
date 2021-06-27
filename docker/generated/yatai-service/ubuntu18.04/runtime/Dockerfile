# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM yatai-service:base-python$PYTHON_VERSION-ubuntu18.04 as build_image

RUN pip install bentoml[yatai_service]==$BENTOML_VERSION --no-cache-dir && \
    apt-get purge --autoremove -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY tools/yatai-service/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "yatai-service-start"]
