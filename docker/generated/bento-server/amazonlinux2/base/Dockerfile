# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================


ARG XX_VERSION=1.1.0

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

ARG PYTHON_VERSION

FROM amazonlinux:2 as python

WORKDIR /tmp

RUN <<"EOT"
if [ "${PYTHON_VERSION}" == "3.6" ]; then
	FULL_VERSION="3.6.15"
elif [ "${PYTHON_VERSION}" == "3.7" ]; then
	FULL_VERSION="3.7.13"
elif [ "${PYTHON_VERSION}" == "3.8" ]; then
	FULL_VERSION="3.8.13"
elif [ "${PYTHON_VERSION}" == "3.9" ]; then
	FULL_VERSION="3.9.11"
elif [ "${PYTHON_VERSION}" == "3.10" ]; then
	FULL_VERSION="3.10.3"
fi
yum -y update
yum groupinstall "Development Tools" && \
	yum -y install \
		gcc openssl11 openssl11-devel bzip2-devel libffi-devel wget curl

curl -O https://bootstrap.pypa.io/get-pip.py
wget https://www.python.org/ftp/python/${FULL_VERSION}/Python-${FULL_VERSION}.tgz && tar xvf Python-${FULL_VERSION}.tgz
cd Python-${PYTHON_VERSION}*/ && ./configure --enable-optimizations
make -j$(nproc) install
EOT

FROM amazonlinux:2 as base

COPY --from=xx / /
COPY --from=python /usr /usr
COPY --from=python /lib /lib
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

ENV PATH "$PATH:$HOME/.local/bin"

RUN python3 get-pip.py --user

RUN --mount=type=cache,target=/var/cache/yum yum upgrade -y \
		&& yum install -y gcc gcc-c++ make \
		&& yum clean all