# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM yatai-service:base-python$PYTHON_VERSION-centos7 as build_image

ENV BENTOML_VERSION=0.13.0

RUN pip install bentoml[yatai_service]==${BENTOML_VERSION} --no-cache-dir --use-deprecated=legacy-resolver

COPY tools/yatai-service/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "yatai-service-start"]
