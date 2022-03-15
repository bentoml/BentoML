ARG XX_VERSION=1.1.0

FROM --platform=$TARGETARCH tonistiigi/xx:${XX_VERSION} AS xx

FROM --platform=$TARGETARCH tonistiigi/bats-assert AS assert

FROM --platform=$TARGETARCH tonistiigi/alpine as test

COPY --from=assert . .
COPY --from=xx / /

WORKDIR /workspace

RUN xx-apk add --no-cache bats

FROM test
