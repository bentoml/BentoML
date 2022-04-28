# syntax = docker/dockerfile:1.3-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================

ARG PYTHON_VERSION

ARG XX_VERSION=1.1.0

ARG RELEASE_DISTROS=alpine3.14

FROM --platform=$BUILDPLATFORM tonistiigi/xx:${XX_VERSION} AS xx

FROM --platform=$BUILDPLATFORM python:${PYTHON_VERSION}-${RELEASE_DISTROS} as base

FROM base as base-amd64

FROM base as base-arm64

FROM base as base-ppc64le

FROM base as base-s390x

FROM base-${TARGETARCH}

COPY --from=xx / /

ARG TARGETPLATFORM

ARG TARGETARCH

ENV PATH /usr/local/bin:$PATH

ENV ENV /root/.bashrc

CMD [ "/bin/bash" ]

ARG PYTHON_VERSION

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8

ENV PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apk xx-apk add --update bash gcc libc-dev shadow