#!/usr/bin/env bats

load 'helpers'
load 'assert'

image="${BENTOML_TEST_ORGANIZATION}/${BENTOML_TEST_IMAGE_NAME}:${BENTOML_TEST_BENTOML_VERSION}-python${BENTOML_TEST_PYTHON_VERSION}-${BENTOML_TEST_OS}-${BENTOML_TEST_IMAGE_TAG_PREFIX}"

function setup() {
    DEBUG=1
    setup_general ${BENTOML_TEST_ARCH} ${image}
}

function teardown() {
    cleanup
}

@test "bentoml_version" {
    docker_run ${image} bentoml --version
	assert_success
}

@test "conda_version" {
    docker_run ${image} conda --version
	assert_success
}

@test "bash_version" {
    docker_run ${image} bash --version
	assert_success
}

@test "python_version" {
    docker_run ${image} python --version
	assert_success
}
