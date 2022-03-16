#!/bin/bash

# Check if we're in rootless mode.
ROOTLESS=$(id -u)

ARCH=$(uname -m)

# check for nividia-docker driver
HAS_GPU=$(docker info | grep Runtimes | grep -q 'nvidia')
GPU_SUPPORTED_OS=( debian11 debian10 ubi8 ubi7 )

# make sure that we are using runc
RUNTIME="$(docker info | grep Runtimes)"

function debug() {
    if [[ $DEBUG -eq 1 ]]; then
        echo "DEBUG: $@" | sed -e 's/^/# /' >&3 ;
    fi
}

function check_runc() {
    if ! $(echo"$RUNTIME" | grep 'runc' &>/dev/null ); then
		fail "Required runc as container runtime."
    fi
    debug "$RUNTIME"
}

function cleanup() {
    # Check caller.
	if batslib_is_caller --indirect 'teardown'; then
		echo "Must be called from \`teardown'" \
		batslib_decorate 'ERROR: clean_up' \
		fail
		return $?
	fi

    echo "doing some cleanup..."
    run unset "${!BENTOML_TEST@}"
}

function setup_general() {
	requires root nividialess
	PLATFORM = "$1"
    IMG = "$2"

    docker pull --platform "$PLATFORM" "$IMG"

	if [[ $DEBUG -eq 1 ]]; then
		check_runc
    fi
}

function setup_gpu(){
	requires root nividia
	PLATFORM = "$1"
	IMG = "$2"

    docker pull --platform "$PLATFORM" "$IMG"
}


function requires() {
	for var in "$@"; do
		local skip_me
		case $var in
		root)
			if [ "$ROOTLESS" -ne 0 ]; then
				skip_me=1
			fi
			;;
		rootless)
			if [ "$ROOTLESS" -eq 0 ]; then
				skip_me=1
			fi
			;;
		nvidia)
			if [ "$HAS_GPU" -ne 0 ]; then
				skip_me=1
			fi
			;;
		nvidialess)
			if [ "$HAS_GPU" -eq 0 ]; then
				skip_me=1
			fi
			;;
		supports_gpu)
		    if [[ "${GPU_SUPPORTED_OS[*]}" =~ "${BENTOML_TEST_OS}" ]]; then
				skip_me=1
			fi
			;;
		arch_x86_64)
			if [ "$ARCH" != "x86_64" ]; then
				skip_me=1
			fi
			;;
		arch_arm64)
			if [ "$ARCH" != "aarch64" ] || [ "$ARCH" != 'arm64' ]; then
				skip_me=1
			fi
			;;
		*)
			fail "BUG: Invalid requires $var."
			;;
		esac
		if [ -n "$skip_me" ]; then
			skip "test requires $var"
		fi
	done
}


# Taken from runc tests
function docker_run() {
    run docker run --privileged --rm --init "$@"
    echo "docker run $@ (status=$status):" >&2
    echo "$output" >&2
}

function docker_run_arch() {
    run docker run --privileged --rm --init --platform linux/"$ARCH" "$@"
    echo "docker run $@ (status=$status):" >&2
    echo "$output" >&2
}

function docker_pull() {
    run docker pull --platform linux/"$ARCH" "$@"
    echo "docker pull $@ (status=$status):" >&2
    echo "$output" >&2
}

function docker_rmi() {
    run docker rmi -f "$@"
    echo "docker rmi $@ (status=$status):" >&2
    echo "$output" >&2
}
