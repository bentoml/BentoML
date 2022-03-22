#!/usr/bin/env bats

load 'helpers'
load 'assert'

function setup() {
    setup_general ${BENTOML_TEST_ARCH} ${IMAGE}
}

function teardown() {
    cleanup
}

@test "check_architecture" {
    docker_run_arch ${BENTOML_TEST_ARCH} --env narch=${BENTOML_TEST_ARCH} ${IMAGE} bash -c '[[ "$(uname -m)" == "${narch}" ]] || false'
    assert_success
}
