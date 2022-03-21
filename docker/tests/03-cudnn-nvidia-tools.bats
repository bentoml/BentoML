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

@test "check_nvcc_installed" {
    docker_run --gpus 0 ${image} bash -c "stat /usr/local/cuda/bin/nvcc"
	assert_success
}

@test "check_gcc_installed" {
    docker_run --gpus 0 ${image} bash -c "gcc --version"
	assert_success
}


