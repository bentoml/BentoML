# Docker builder images

ARG XX_VERSION=1.1.0

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

FROM --platform=$BUILDPLATFORM docker:latest

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

ENV DOCKER_TLS_CERTDIR "/certs"

ENV DOCKER_CLI_EXPERIMENTAL enabled

ENV DOCKER_BUILDKIT=1 

WORKDIR /bentoml

VOLUME ["/bentoml"]

ENV DOCKER_BUILDKIT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="$HOME/pypoetry" \
    POETRY_HOME="$HOME/.local" \
    PATH="${PATH}:$HOME/.local/bin"

RUN xx-apk add --no-cache wget git bash findutils python3 python3-dev curl g++ libmagic skopeo jq make

ENV PUSHRM_URL https://github.com/christian-korneck/docker-pushrm/releases/download/v1.8.0/docker-pushrm_linux_

ENV BUILDX_URL https://github.com/docker/buildx/releases/download/v0.7.0/buildx-v0.7.0.linux-

RUN mkdir -p $HOME/.docker/cli-plugins/

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" == "x86_64" ]; then \
        PUSHRM_ARCH='amd64'; \
        BUILDX_ARCH='amd64'; \
    elif [ "${UNAME_M}" == "aarch64" ]; then \
        PUSHRM_ARCH='arm64'; \
        BUILDX_ARCH='arm64'; \
    elif [ "${UNAME_M}" == "armv7l" ]; then \
        PUSHRM_ARCH='arm'; \
        BUILDX_ARCH='arm-v7'; \
    elif [ "${UNAME_M}" == "armv6l" ]; then \
        PUSHRM_ARCH='arm'; \
        BUILDX_ARCH='arm-v6'; \
    else \
        echo "couldn't find a version for ${UNAME_M} that supports both docker-pushrm and buildx"; \
        exit 1; \
    fi && \
    wget -O $HOME/.docker/cli-plugins/docker-pushrm $PUSHRM_URL${PUSHRM_ARCH} && \
    wget -O $HOME/.docker/cli-plugins/docker-buildx $BUILDX_URL${BUILDX_ARCH}

RUN chmod a+x $HOME/.docker/cli-plugins/docker-pushrm && \
    chmod a+x $HOME/.docker/cli-plugins/docker-buildx

RUN if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi

RUN if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3 /usr/bin/python; fi

RUN python3 -m ensurepip

RUN rm -r /usr/lib/python*/ensurepip

RUN pip3 install --upgrade pip setuptools

LABEL maintainer="BentoML Team <contact@bentoml.com>"

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

COPY pyproject.toml .

RUN poetry install

COPY hack/bashrc /etc/bash.bashrc

RUN chmod a+rwx /etc/bash.bashrc

RUN echo "source /etc/bash.bashrc" >> $HOME/.bashrc

CMD [ "bash" ]
