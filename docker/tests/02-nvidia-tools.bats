#!/usr/bin/env bats

load helpers

GPU_SUPPORTED_OS=( debian11 debian10 ubi8 ubi7 )

image="bentoml/${__TEST_IMAGE_NAME}:${__TEST_BENTOML_VERSION}-python${__TEST_PYTHON_VERSION}-${__TEST_OS}-cudnn"

function setup() {
    # docker pull --platform linux/${__TEST_ARCH} ${image}
    if [[ "${GPU_SUPPORTED_OS[*]}" =~ "${__TEST_OS}" ]]; then
        docker pull ${image}
        check_gpu
    else
        :
    fi
}

function teardown() {
    cleanup
}

@test "check_nvcc_installed" {
    docker_run --rm --gpus 0 ${image} bash -c "stat /usr/local/cuda/bin/nvcc"
    [ "$status" -eq 0 ]
}

@test "check_gcc_installed" {
    docker_run --rm --gpus 0 ${image} bash -c "gcc --version"
    [ "$status" -eq 0 ]
}


