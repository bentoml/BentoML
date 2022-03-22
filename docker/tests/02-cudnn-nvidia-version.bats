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

@test "check_if_nvidia_is_available" {
    docker_run --gpus 0 ${IMAGE} nvidia-smi
	assert_success
}

@test "check_LD_LIBRARY_PATH" {
    docker_run --gpus 0 ${IMAGE} bash -c "printenv | grep -q 'LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64'"
	assert_success
}

