# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


ARG OS_VERSION

FROM ubuntu:${OS_VERSION} as base-image

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
        ca-certificates gnupg2 wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

ENV PYTHONDONTWRITEBYTECODE=1

# we will install python from conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} pip \
    && /opt/conda/bin/conda clean -afy

FROM ubuntu:${OS_VERSION} as build-image

# Redefine BENTOML_VERSION because of multistage
ARG BENTOML_VERSION

COPY --from=base-image /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH

RUN pip install bentoml[model_server]==${BENTOML_VERSION} --no-cache-dir

COPY tools/model-server/entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]

CMD ["bentoml", "serve-gunicorn", "/bento"]