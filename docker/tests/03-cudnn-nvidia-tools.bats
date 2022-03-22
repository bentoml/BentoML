#!/usr/bin/env bats

load 'helpers'
load 'assert'

function setup() {
	requires supports_gpu
	setup_gpu ${BENTOML_TEST_ARCH} ${IMAGE}
}

function teardown() {
    cleanup
}

@test "check_nvcc_installed" {
    docker_run --gpus 0 ${IMAGE} bash -c "stat /usr/local/cuda/bin/nvcc"
	assert_success
}

@test "check_gcc_installed" {
    docker_run --gpus 0 ${IMAGE} bash -c "gcc --version"
	assert_success
}


