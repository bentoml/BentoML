# syntax = docker/dockerfile:1.3-labs

FROM koalaman/shellcheck-alpine:latest as shellcheck
WORKDIR /src
COPY hack/shells/ .
RUN shellcheck *

FROM mvdan/shfmt:latest-alpine as shfmt

FROM jessfraz/dockfmt:latest as dockfmt
