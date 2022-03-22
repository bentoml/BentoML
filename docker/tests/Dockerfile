# syntax = docker/dockerfile:1.3-labs

ARG TEST_TYPE=runtime

ARG XX_VERSION=1.1.0
ARG ALPINE_VERSION=3.15
ARG DOCKERD_VERSION=20.10.13

FROM tonistiigi/xx:${XX_VERSION} AS xx

FROM tonistiigi/bats-assert AS assert

FROM --platform=$BUILDPLATFORM docker:${DOCKERD_VERSION}-alpine${ALPINE_VERSION} as base

WORKDIR /work

COPY --from=assert . .
COPY --from=xx / /

RUN --mount=type=cache,target=/pkg-cache \
    ln -s /pkg-cache /etc/apk/cache && \
    xx-apk add --no-cache bats vim

FROM base as bats-test

COPY ./run_tests.sh /work
COPY ./helpers.bash /work

FROM bats-test as bats-runtime

COPY ./00-runtime-check-arch.bats /work/00-check-arch.bats
COPY ./01-runtime-check-requirements.bats /work/01-requirements.bats

FROM bats-test as bats-cudnn

COPY ./00-runtime-check-arch.bats /work/00-check-arch.bats
COPY ./02-cudnn-nvidia-version.bats /work/01-nvidia-version.bats
COPY ./03-cudnn-nvidia-tools.bats /work/02-nvidia-tools.bats

FROM bats-${TEST_TYPE} as test

ARG TEST_TYPE

WORKDIR /work

SHELL ["/bin/bash", "-euo", "pipefail", "-c" ]

CMD [ "./run_tests.sh", "-default" ]

FROM test as test-amd64

FROM test as test-arm64

FROM test as test-arm

FROM test-${TARGETARCH}

ARG TARGETARCH
