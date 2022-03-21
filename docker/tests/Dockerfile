# syntax=docker/dockerfile:1.3-labs

ARG TEST_BASE_DISTROS=alpine
ARG ORGANIZATION=aarnphm

FROM --platform=$BUILDPLATFORM ${ORGANIZATION}/bats-test as base

FROM base as base-alpine

FROM base as base-debian

FROM base as base-rhel

FROM base as base-ami2
