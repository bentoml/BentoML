ARG PYTHON_VERSION
ARG BENTOML_VERSION

ENV PATH /opt/conda/bin:$PATH

ENV PYTHONDONTWRITEBYTECODE=1

# we will install python from conda.
RUN apk add --no-cache --virtual .build-dependencies ca-certificates wget \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -f -b -p /opt/conda \
    && conda update --all --yes \
    && conda config --set auto_update_conda False \
    && apk del --purge .build-dependencies \
    && rm -f ~/miniconda.sh \
    && conda clean --all --force-pkgs-dirs --yes \
    && find /opt/conda/ -follow -type f \( -iname '*.a' -o -iname '*.pyc' -o -iname '*.js.map' \) -delete

# bash is needed to run bentoml.
RUN conda install -y python=${PYTHON_VERSION} pip \
    && conda clean -afy \
    && apk add --no-cache bash