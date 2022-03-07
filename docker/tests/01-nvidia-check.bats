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

@test "check_if_nvidia_is_available" {
    if [[ "${GPU_SUPPORTED_OS[*]}" =~ "${__TEST_OS}" ]]; then
        skip "Image doesn't support GPU"
    fi
    docker_run --rm --gpus 0 ${image} nvidia-smi
    [ "$status" -eq 0 ]
}

@test "check_LD_LIBRARY_PATH" {
    if [[ "${GPU_SUPPORTED_OS[*]}" =~ "${__TEST_OS}" ]]; then
        skip "Image doesn't support GPU"
    fi
    docker_run --rm --gpus 0 ${image} bash -c "printenv | grep -q 'LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64'"
    [ "$status" -eq 0 ]
}

