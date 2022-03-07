#!/usr/bin/env bats

load helpers

image="bentoml/${__TEST_IMAGE_NAME}:${__TEST_BENTOML_VERSION}-python${__TEST_PYTHON_VERSION}-${__TEST_OS}-runtime"

function setup() {
    # docker pull --platform linux/${__TEST_ARCH} ${image}
    docker pull --platform linux/amd64 ${image}
    check_runtime
}

function teardown() {
    cleanup
}

@test "check_architecture" {
    narch=${__TEST_ARCH}
    if [[ ${__TEST_ARCH} == "arm64" ]]; then
        narch="aarch64"
    fi
    docker_run --rm --env narch=${narch} --platform linux/${__TEST_ARCH} ${image} bash -c '[[ "$(uname -m)" == "${narch}" ]] || false'
    [ "$status" -eq 0 ]
}
