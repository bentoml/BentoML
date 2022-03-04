# syntax = docker/dockerfile:1.2

FROM python:3.10-slim

LABEL maintainer="BentoML Team <contact@bentoml.ai>"

ENV DOCKER_BUILDKIT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="/var/cache/pypoetry" \
    PATH="$PATH:/root/.local/bin"

RUN apt-get update && \
    apt-get install -y bash curl build-essential \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

# https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/
RUN curl -sSL https://get.docker.com/ | sh

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /bentoml

VOLUME ["/bentoml"]

COPY pyproject.toml .

COPY tools/bashrc /etc/bash.bashrc

COPY --from=docker/buildx-bin:latest /buildx /usr/libexec/docker/cli-plugins/docker-buildx

RUN chmod a+rwx /etc/bash.bashrc

RUN if [ ! -e /usr/bin/pip ]; then \
        ln -s /usr/bin/pip3 /usr/bin/pip; \
    fi

RUN if [ ! -e /usr/bin/python ]; then \
        ln -sf /usr/bin/python3 /usr/bin/python; \
    fi

RUN python -m pip install --upgrade pip setuptools

RUN poetry install

CMD [ "bash", "-c" ]
