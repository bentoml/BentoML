# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM yatai-service:python3.6-ubuntu20.04-base

RUN pip install bentoml[yatai_service]==$BENTOML_VERSION --no-cache-dir && \
    apt-get purge --autoremove -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY tools/yatai-service/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "yatai-service-start"]
