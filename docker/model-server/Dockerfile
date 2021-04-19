FROM continuumio/miniconda3:4.9.2

RUN apt-get update --fix-missing && \
    apt-get install -y build-essential && \
    apt-get clean

RUN conda update -n base -c defaults conda

ARG BENTOML_VERSION
ENV BENTOML_VERSION=$BENTOML_VERSION

ARG PYTHON_VERSION
ENV PYTHON_VERSION=$PYTHON_VERSION

## install proper python major version
RUN conda install pip python=$PYTHON_VERSION -y

# pre-install BentoML base dependencies
RUN pip install bentoml[model_server]==$BENTOML_VERSION --no-cache-dir

COPY entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]
CMD ["bentoml", "serve-gunicorn", "/bento"]
