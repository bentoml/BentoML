#!/bin/bash

# Check if we're in rootless mode.
ROOTLESS=$(id -u)

# check for nividia-docker driver
HAS_GPU=0
GPU_SUPPORTED_OS=(debian11 debian10 ubi8 ubi7)

function debug() {
	if [[ $DEBUG -eq 1 ]]; then
		echo "DEBUG: " "${@}" | sed -e 's/^/# /' >&3
	fi
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
	run unset "${IMAGE}"
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
			if [ "$HAS_GPU" -eq 0 ]; then
				skip_me=1
			fi
			;;
		supports_gpu)
			if [[ "${GPU_SUPPORTED_OS[*]}" =~ ${BENTOML_TEST_OS} ]]; then
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
	echo "docker run --privileged --rm --init $@ (status=$status):" >&2
	echo "$output" >&2
}

function docker_run_arch() {
	ARCH="$1"
	shift
	run docker run --privileged --rm --init --platform "linux/$ARCH" "$@"
	echo "docker run --privileged --rm --init --platform linux/$ARCH $@ (status=$status):" >&2
	echo "$output" >&2
}

function docker_pull() {
	ARCH="$1"
	shift
	run docker pull --platform "linux/$ARCH" "$@"
	echo "docker pull --platform linux/$ARCH $@ (status=$status):" >&2
	echo "$output" >&2
}

function docker_rmi() {
	run docker rmi -f "$@"
	echo "docker rmi -f $@ (status=$status):" >&2
	echo "$output" >&2
}

function setup_general() {
	requires root
	docker_pull "$1" "$2"
}

function setup_gpu() {
	requires root nvidia
	docker_pull "$1" "$2"
}
