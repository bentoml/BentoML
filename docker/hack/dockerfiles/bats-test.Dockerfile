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

FROM base as test

FROM base as test-amd64

FROM base as test-arm64

FROM base as test-arm

FROM test-${TARGETARCH}

ARG TARGETARCH
