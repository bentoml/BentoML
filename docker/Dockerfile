# syntax = docker/dockerfile:1.2

FROM debian:buster-slim

LABEL maintainer="BentoML Contributor <contact@bentoml.ai>"

ENV DOCKER_BUILDKIT=1

RUN apt-get update && \
    apt-get install -y python3 python3-pip bash curl build-essential

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

# https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/
RUN curl -sSL https://get.docker.com/ | sh

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3

WORKDIR /bentoml

VOLUME ["/bentoml"]

COPY pyproject.toml .

COPY tools/bashrc /etc/bash.bashrc

RUN chmod a+rwx /etc/bash.bashrc

RUN if [ ! -e /usr/bin/pip ]; then \
        ln -s pip3 /usr/bin/pip; \
    fi

RUN if [ ! -e /usr/bin/python ]; then \
        ln -sf /usr/bin/python3 /usr/bin/python; \
    fi

RUN python -m pip install --upgrade pip setuptools

RUN . $HOME/.poetry/env && \
    poetry config virtualenvs.create false && \
    poetry install