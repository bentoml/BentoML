ARG XX_VERSION=1.1.0
ARG ALPINE_VERSION=3.15
ARG DOCKERD_VERSION=20.10.13

FROM tonistiigi/xx:${XX_VERSION} AS xx

FROM tonistiigi/bats-assert AS assert

FROM --platform=$BUILDPLATFORM docker:${DOCKERD_VERSION}-alpine${ALPINE_VERSION} as base

COPY --from=assert . .
COPY --from=xx / /

WORKDIR /work

RUN xx-apk add --no-cache bats

COPY ./tests ./test

COPY hack/runt .

FROM base as test

COPY --from=base / /

COPY --from=base / /

FROM base as test-amd64

FROM base as test-arm64

FROM base as test-arm

FROM test-${TARGETARCH}

ARG TARGETARCH
