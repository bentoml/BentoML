# syntax = docker/dockerfile:1.3-labs

ARG TEST_TYPE=runtime

ARG ORGANIZATION
ARG PYTHON_VERSION
ARG BENTOML_VERSION

FROM --platform=$BUILDPLATFORM ${ORGANIZATION}/bats-test as test

COPY ./run_tests.sh /work
COPY ./helpers.bash /work

FROM test as test-runtime

COPY ./00-runtime-check-arch.bats /work/00-check-arch.bats
COPY ./01-runtime-check-requirements.bats /work/01-check-requirements.bats

FROM test as test-cudnn

COPY ./02-cudnn-nvidia-version.bats /work/00-check-nvidia-version.bats
COPY ./03-cudnn-nvidia-tools.bats /work/01-check-nvidia-tools.bats

FROM test-${TEST_TYPE}

ARG TEST_TYPE

WORKDIR /work

FROM test-${TEST_TYPE} as test-debian11

ARG TEST_TYPE

SHELL ["/bin/bash", "-euo", "pipefail", "-c" ]

RUN --mount=type=cache,target=/pkg-cache \
	./run_tests.sh -default -python_version ${PYTHON_VERSION} -distros debian11 -suffix ${TEST_TYPE} -organization ${ORGANIZATION} -bentoml_version ${BENTOML_VERSION}

FROM test-${TEST_TYPE} as test-debian10

ARG TEST_TYPE

RUN --mount=type=cache,target=/pkg-cache \
	./run_tests.sh -default -python_version ${PYTHON_VERSION} -distros debian10 -suffix ${TEST_TYPE} -organization ${ORGANIZATION} -bentoml_version ${BENTOML_VERSION}

FROM test-${TEST_TYPE} as test-alpine

ARG TEST_TYPE

RUN --mount=type=cache,target=/pkg-cache \
	./run_tests.sh -default -python_version ${PYTHON_VERSION} -distros alpine3.14 -suffix ${TEST_TYPE} -organization ${ORGANIZATION} -bentoml_version ${BENTOML_VERSION}

FROM test-${TEST_TYPE} as test-ubi8

ARG TEST_TYPE

RUN --mount=type=cache,target=/pkg-cache \
	./run_tests.sh -default -python_version ${PYTHON_VERSION} -distros ubi8 -suffix ${TEST_TYPE} -organization ${ORGANIZATION} -bentoml_version ${BENTOML_VERSION}

FROM test-${TEST_TYPE} as test-ami2

ARG TEST_TYPE

RUN --mount=type=cache,target=/pkg-cache \
	./run_tests.sh -default -python_version ${PYTHON_VERSION} -distros amazonlinux2 -suffix ${TEST_TYPE} -organization ${ORGANIZATION} -bentoml_version ${BENTOML_VERSION}
