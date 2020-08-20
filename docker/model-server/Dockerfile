FROM continuumio/miniconda3:4.8.2

RUN apt-get update --fix-missing && \
    apt-get install -y build-essential && \
    apt-get clean

ARG BENTOML_VERSION
ENV BENTOML_VERSION=$BENTOML_VERSION

# pre-install BentoML base dependencies
RUN pip install bentoml==$BENTOML_VERSION

COPY entrypoint.sh /usr/local/bin/

ENTRYPOINT [ "entrypoint.sh" ]
CMD ["bentoml", "serve-gunicorn", "/bento"]
