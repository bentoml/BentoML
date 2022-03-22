#!/usr/bin/env bats

load 'helpers'
load 'assert'

function setup() {
    setup_general ${BENTOML_TEST_ARCH} ${IMAGE}
}

function teardown() {
    cleanup
}


@test "bentoml_version" {
    docker_run ${IMAGE} bentoml --version
	assert_success
}

@test "conda_version" {
    docker_run ${IMAGE} conda --version
	assert_success
}

@test "bash_version" {
    docker_run ${IMAGE} bash --version
	assert_success
}

@test "python_version" {
    docker_run ${IMAGE} python --version
	assert_success
}
