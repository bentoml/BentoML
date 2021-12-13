{{ metadata.header }}

ARG PYTHON_VERSION

FROM {{ metadata.package }}:{{ build_tag }} as build_image

ENV BENTOML_VERSION={{ metadata.envars['BENTOML_VERSION'] }}

# bash is needed to run bentoml.
RUN conda install -y python=$PYTHON_VERSION pip \
    && conda clean -afy \
    && apk add --no-cache bash git build-base

RUN pip install bentoml==${BENTOML_VERSION} --no-cache-dir
