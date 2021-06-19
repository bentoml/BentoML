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