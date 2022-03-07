#!/bin/bash

function debug() {
    if [[ $DEBUG -eq 1 ]]; then
        echo "DEBUG: $@" | sed -e 's/^/# /' >&3 ;
    fi
}

function check_runtime() {
    debug "$(docker info | grep Runtimes)"
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
    echo "docker run $@ (status=$status):" >&2
    echo "$output" >&2
}

