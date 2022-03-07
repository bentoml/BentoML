#!/bin/bash

function debug() {
    if [[ $DEBUG -eq 1 ]]; then
        echo "DEBUG: $@" | sed -e 's/^/# /' >&3 ;
    fi
}

function check_runtime() {
    debug "$(docker info | grep Runtimes)"
}

function cleanup() {
    echo "doing some cleanup..."
    run unset __TEST_PYTHON_VERSION
    run unset __TEST_OS
    run unset __TEST_IMAGE_TAG_SUFFIX
    run unset __TEST_BENTOML_VERSION
    run unset __TEST_IMAGE_NAME
    run unset __TEST_ARCH
}


function check_gpu() {
    docker info | grep 'Runtimes:' | grep -q 'nvidia'
}

# Taken from runc tests
function docker_run() {
    run docker run "$@"
    echo "docker run $@ (status=$status):" >&2
    echo "$output" >&2
}

function docker_build() {
    run docker build --pull "$@"
    echo "docker build $@ (status=$status):" >&2
    echo "$output" >&2
}

function docker_rmi() {
    run docker rmi -f "$@"
    echo "docker rmi $@ (status=$status):" >&2
    echo "$output" >&2
}
