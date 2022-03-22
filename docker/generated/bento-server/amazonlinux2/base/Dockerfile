# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG XX_VERSION=1.1.0

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

FROM amazonlinux:2 as python
ARG PYTHON_VERSION

ENV PYTHON_VERSION="${PYTHON_VERSION}"

WORKDIR /tmp

RUN --mount=type=cache,sharing=shared,target=/var/cache/yum \
    yum groupinstall -y "Development Tools" \
    && yum -y install gcc openssl11 openssl11-devel bzip2-devel libffi-devel wget curl gcc-c++ make git \
    && yum clean all

RUN <<"EOT"
set -ex
PIP_URL="https://bootstrap.pypa.io/get-pip.py"
if [ "${PYTHON_VERSION}" = "3.6" ]; then 
    FULL_VERSION="3.6.15"
    PIP_URL="https://bootstrap.pypa.io/pip/3.6/get-pip.py"
elif [ "${PYTHON_VERSION}" = "3.7" ]; then 
    FULL_VERSION="3.7.13"
elif [ "${PYTHON_VERSION}" = "3.8" ]; then 
    FULL_VERSION="3.8.13"
elif [ "${PYTHON_VERSION}" = "3.9" ]; then 
    FULL_VERSION="3.9.11"
elif [ "${PYTHON_VERSION}" = "3.10" ]; then 
    FULL_VERSION="3.10.3"
else 
    echo "${PYTHON_VERSION} is not yet supported"
    exit 1
fi 
curl -O "${PIP_URL}"
wget https://www.python.org/ftp/python/"${FULL_VERSION}"/Python-"${FULL_VERSION}".tgz 
tar xvf Python-"${FULL_VERSION}".tgz 
cd Python-"${FULL_VERSION}"/ && ./configure --enable-optimizations 
make -j4 install
EOT

FROM amazonlinux:2 as base

COPY --from=xx / /

COPY --from=python /usr /usr
COPY --from=python /lib /lib
COPY --from=python /etc /etc
COPY --from=python /tmp/get-pip.py /tmp/get-pip.py

FROM base as base-arm64

FROM base as base-amd64

FROM base-${TARGETARCH}

ARG TARGETARCH

# setup ENV and ARG
ARG PYTHON_VERSION
ENV PYTHONDONTWRITEBYTECODE=1

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

ENV PATH "$PATH:/root/.local/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin"

RUN python3 /tmp/get-pip.py --user

RUN if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi

RUN if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3 /usr/bin/python; fi