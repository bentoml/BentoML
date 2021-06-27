# syntax = docker/dockerfile:experimental
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG PYTHON_VERSION

FROM model-server:python${PYTHON_VERSION}-alpine-base as build_image

ENV ENV /root/.shinit
CMD [ "/bin/sh" ]

ARG PYTHON_VERSION

# we will install python from conda.
RUN apk add --no-cache --virtual .build-dependencies ca-certificates wget \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -f -b -p /opt/conda \
    && conda update --all --yes \
    && conda config --set auto_update_conda False \
    \
    && apk del --purge .build-dependencies \
    && rm -f ~/miniconda.sh \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f \( -iname '*.a' -o -iname '*.pyc' -o -iname '*.js.map' \) -delete

FROM build_image as release_image

ARG BENTOML_VERSION

# bash is needed to run bentoml.
RUN conda install -y python=$PYTHON_VERSION pip \
    && conda clean -afy \
    && apk add --no-cache bash git

RUN pip install bentoml[model_server]==$BENTOML_VERSION --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]
