# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM model-server:python3.6-slim-base

RUN pip install bentoml[model_server]==$BENTOML_VERSION --no-cache-dir && \
    apt-get purge --autoremove -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]
