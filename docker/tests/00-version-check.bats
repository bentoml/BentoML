#!/usr/bin/env bats

load 'helpers'
load 'assert'

image="${__ORGANIZATION}/${__TEST_IMAGE_NAME}:${__TEST_BENTOML_VERSION}-python${__TEST_PYTHON_VERSION}-${__TEST_OS}-${__TEST_IMAGE_TAG_SUFFIX}"

function setup() {
    docker pull --platform linux/${__TEST_ARCH} ${image}
    check_runtime
}

function teardown() {
    cleanup
}

@test "bentoml_version" {
    docker_run --rm ${image} bentoml --version
	assert_success
}

@test "conda_version" {
    docker_run --rm ${image} conda --version
	assert_success
}

@test "bash_version" {
    docker_run --rm ${image} bash --version
	assert_success
}

@test "python_version" {
    docker_run --rm ${image} python --version
	assert_success
}
