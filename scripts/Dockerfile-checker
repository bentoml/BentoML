# syntax = docker/dockerfile:1.2
FROM python:3.8-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1

ENV NODE_VERSION=16.13.0

RUN apt-get update \
    && apt-get install -q -y --no-install-recommends \
        ca-certificates curl git gnupg build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

RUN curl -fsSLO https://github.com/mikefarah/yq/releases/download/v4.14.1/yq_linux_amd64.tar.gz \
    && tar -zvxf ./yq_linux_amd64.tar.gz  \
    && mv ./yq_linux_amd64 /usr/local/bin/yq \
    && rm -f ./yq_linux_amd64.tar.gz ./yq.1 ./install-man-page.sh

RUN npm i -g npm@^7 yarn pyright

RUN python -m pip install -U pip setuptools

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements/dev-requirements.txt .

COPY requirements/tests-requirements.txt .

RUN pip install -r ./dev-requirements.txt

WORKDIR "/bentoml"

VOLUME ["/bentoml"]
