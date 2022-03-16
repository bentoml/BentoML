#!/usr/bin/env bats

load 'helpers'
load 'assert'

image="${BENTOML_TEST_ORGANIZATION}/${BENTOML_TEST_IMAGE_NAME}:${BENTOML_TEST_BENTOML_VERSION}-python${BENTOML_TEST_PYTHON_VERSION}-${BENTOML_TEST_OS}-${BENTOML_TEST_IMAGE_TAG_PREFIX}"

function setup() {
    setup_general ${BENTOML_TEST_ARCH} ${image}
}

function teardown() {
    cleanup
}

@test "check_architecture" {
    narch=${BENTOML_TEST_ARCH}
    docker_run --rm --env narch=${narch} --platform linux/${BENTOML_TEST_ARCH} ${image} bash -c '[[ "$(uname -m)" == "${narch}" ]] || false'
    assert_success
}
