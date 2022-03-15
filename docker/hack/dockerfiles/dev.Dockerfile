#syntax=docker/dockerfile:1.3-labs

ARG XX_VERSION=1.1.0

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

FROM python:3.10.2-alpine3.15 as base_build

# this next part is ported from docker/docker
RUN apk add --no-cache \
		ca-certificates \
		libc6-compat \
		openssh-client \
        bash

ENV DOCKER_VERSION 20.10.13

RUN [ ! -e /etc/nsswitch.conf ] && echo 'hosts: files dns' > /etc/nsswitch.conf

RUN bash <<"EOT"
set -eux
apkArch="$(apk --print-arch)";

case "$apkArch" in
    "x86_64")
        url="https://download.docker.com/linux/static/stable/x86_64/docker-20.10.13.tgz";
        ;;
    "armhf")
        url="https://download.docker.com/linux/static/stable/armel/docker-20.10.13.tgz";
        ;;
    "armv7")
        url="https://download.docker.com/linux/static/stable/armhf/docker-20.10.13.tgz";
        ;;
    "aarch64")
        url="https://download.docker.com/linux/static/stable/aarch64/docker-20.10.13.tgz";
        ;;
    *) echo >&2 "error: unsupported architecture ($apkArch)"; exit 1 ;;
esac;

wget -O docker.tgz "$url";

tar --extract \
    --file docker.tgz \
    --strip-components 1 \
    --directory /usr/local/bin/;

rm docker.tgz;

dockerd --version
docker --version
EOT

COPY ./hack/dockerfiles/modprobe.sh /usr/local/bin/modprobe
COPY ./hack/dockerfiles/docker-entrypoint.sh /usr/local/bin/

# https://github.com/docker-library/docker/pull/166
ENV DOCKER_TLS_CERTDIR=/certs

RUN mkdir /certs /certs/client && chmod 1777 /certs /certs/client

ENTRYPOINT ["docker-entrypoint.sh"]

CMD [ "sh" ]

FROM base_build as base

FROM base as base-amd64

FROM base as base-arm64

FROM base-${TARGETARCH} as releases

ARG TARGETPLATFORM

ARG TARGETARCH

COPY --from=xx / /

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

ENV BUILDX_URL https://github.com/docker/buildx/releases/download/v0.8.0/buildx-v0.8.0.linux-

RUN mkdir -p $HOME/.docker/cli-plugins/

RUN xx-apk add --no-cache wget git bash findutils readline build-base \
        python3 python3-dev curl gcc libmagic jq make

RUN <<"EOF" bash
set -x
UNAME_M="$(uname -m)"
if [ "${UNAME_M}" == "x86_64" ]; then
    BUILDX_ARCH='amd64';
elif [ "${UNAME_M}" == "aarch64" ]; then
    BUILDX_ARCH='arm64';
elif [ "${UNAME_M}" == "armv7l" ]; then
    BUILDX_ARCH='arm-v7';
elif [ "${UNAME_M}" == "armv6l" ]; then
    BUILDX_ARCH='arm-v6';
elif [ "${UNAME_M}" == "ppc64le" ]; then
    BUILDX_ARCH='ppc64le';
elif [ "${UNAME_M}" == "s390x" ]; then
    BUILDX_ARCH='s390x';
else
    echo "couldn't find a version for ${UNAME_M} that supports buildx";
    exit 1;
fi
wget -O $HOME/.docker/cli-plugins/docker-buildx $BUILDX_URL${BUILDX_ARCH}
EOF

RUN chmod a+x $HOME/.docker/cli-plugins/docker-buildx

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

FROM releases
