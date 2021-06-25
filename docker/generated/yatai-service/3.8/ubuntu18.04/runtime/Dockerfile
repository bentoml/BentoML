# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================



FROM yatai-service:python3.8-ubuntu18.04-base

RUN pip install bentoml[yatai_service]==${BENTOML_VERSION} --no-cache-dir

COPY tools/yatai-service/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "yatai-service-start"]
