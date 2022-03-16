#!/usr/bin/env bats

load 'helpers'
load 'assert'

image="${BENTOML_TEST_ORGANIZATION}/${BENTOML_TEST_IMAGE_NAME}:${BENTOML_TEST_BENTOML_VERSION}-python${BENTOML_TEST_PYTHON_VERSION}-${BENTOML_TEST_OS}-${BENTOML_TEST_IMAGE_TAG_PREFIX}"

function setup() {
    requires supports_gpu
	setup_gpu ${BENTOML_TEST_ARCH} ${image} 
}

function teardown() {
    cleanup
}

@test "check_if_nvidia_is_available" {
    docker_run --gpus 0 ${image} nvidia-smi
	assert_success
}

@test "check_LD_LIBRARY_PATH" {
    docker_run --gpus 0 ${image} bash -c "printenv | grep -q 'LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64'"
	assert_success
}

