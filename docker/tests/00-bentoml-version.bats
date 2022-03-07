#!/usr/bin/env bats

# https://bats-core.readthedocs.io/en/stable/tutorial.html

load helpers

image="${IMAGE_NAME}:${BENTOML_VERSION}-python${PYTHON_VERSION}-${IMAGE_TAG_SUFFIX}"

function setup() {
    docker pull --platform linux/${ARCH} ${image}
    check_runtime
}

function teardown() {
    cleanup
}

@test "nvidia-smi" {
    docker_run --rm --gpus 0 ${image} nvidia-smi
    [ "$status" -eq 0 ]
}
