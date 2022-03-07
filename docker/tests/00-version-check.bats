#!/usr/bin/env bats

load helpers

image="bentoml/${__TEST_IMAGE_NAME}:${__TEST_BENTOML_VERSION}-python${__TEST_PYTHON_VERSION}-${__TEST_OS}-${__TEST_IMAGE_TAG_SUFFIX}"

function setup() {
    # docker pull --platform linux/${__TEST_ARCH} ${image}
    docker pull --platform linux/amd64 ${image}
    check_runtime
}

function teardown() {
    cleanup
}

@test "bentoml_version" {
    docker_run --rm  ${image} bentoml --version
    [ "$status" -eq 0 ]
}

@test "conda_version" {
    docker_run --rm  ${image} conda --version
    [ "$status" -eq 0 ]
}

@test "bash_version" {
    docker_run --rm  ${image} bash --version
    [ "$status" -eq 0 ]
}
