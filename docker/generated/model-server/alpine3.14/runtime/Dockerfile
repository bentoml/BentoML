# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM model-server:base-python$PYTHON_VERSION-alpine as build_image

ENV ENV /root/.shinit
CMD [ "/bin/sh" ]

# bash is needed to run bentoml.
RUN conda install -y python=$PYTHON_VERSION pip \
    && conda clean -afy \
    && apk add --no-cache bash git

RUN pip install bentoml[model_server]==$BENTOML_VERSION --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]
