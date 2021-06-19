# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

FROM debian:${OS_VERSION} as base-image

RUN apt-get update -y \
    && apt-get install -y -q --no-install-recommends build-essential gcc ca-certificates

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

# we will install python from conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -ya

FROM debian:${OS_VERSION} as build-image

COPY --from=base-image /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH

RUN pip install bentoml[model-server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]