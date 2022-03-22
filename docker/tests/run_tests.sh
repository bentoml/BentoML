#!/usr/bin/env bash

set -ex

TEST_PATH="$(realpath $(cd "$(dirname "$(readlink -f "$0")")" && pwd))"
POSITIONAL_ARGS=()

usage() {
	cat <<-EOF
		Usage: run_tests.sh [--help] [image_name][bentoml_version][python_version][distros][suffix][platforms][organization][default] OPTS
		  Some helpers functions for interacting with buildx
		  -h|--help|-help:
		      display this usage.
		  python_version|-python_version|--python_version:
		      Set python version.
		  distros|-distros|--distros:
		      Set distros to test on.
		  bentoml_version|-bentoml_version|--bentoml_version:
		      Set bentoml version.
		  image_name|-image_name|--image_name:
		      Set image name, default to \`bento-server\`
		  suffix|-suffix|--suffix:
		      Set suffix, default to \`runtime\`.
		  organization|-organization|--organization:
		      Set organization, default to \`bentoml\`.
		  platforms|-platforms|--platforms:
		      Set platforms, default to \`amd64\`.

		To run tests locally, make sure to install bats, bats-assert, bats-support. Otherwise use the Docker image.
			./run-tests.sh -default -python_version 3.8 -distros alpine3.14
	EOF
}

while [[ $# -gt 0 ]]; do
	case $1 in
	-h | -help | --help)
		usage
		exit
		;;
	python_version | -python_version | --python_version)
		PYTHON_VERSION="$2"
		shift # past argument
		shift # past value
		;;
	distros | -distros | --distros)
		OS="$2"
		shift # past argument
		shift # past value
		;;
	default | -default | --default)
		PLATFORMS='amd64'
		ORGANIZATION="bentoml"
		IMAGE_NAME='bento-server'
		IMAGE_TAG_SUFFIX="runtime"
		shift # past argument
		;;
	suffix | -suffix | --suffix)
		IMAGE_TAG_SUFFIX="$2"
		shift # past argument
		shift # past value
		;;
	organization | -organization | --organization)
		ORGANIZATION="$2"
		shift # past argument
		shift # past value
		;;
	platforms | -platforms | --platforms)
		PLATFORMS="$2"
		shift # past argument
		shift # past value
		;;
	image_name | -image_name | --image_name)
		IMAGE_NAME="$2"
		shift # past argument
		shift # past value
		;;
	bentoml_version | -bentoml_version | --bentoml_version)
		BENTOML_VERSION="$2"
		shift # past argument
		shift # past value
		;;
	-*)
		echo "Unknown option $1"
		exit 1
		;;
	*)
		POSITIONAL_ARGS+=("$1") # save positional arg
		shift                   # past argument
		;;
	esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

main() {
	export BENTOML_TEST_ARCH="$PLATFORMS"
	image="${ORGANIZATION}/${IMAGE_NAME}:${BENTOML_VERSION}-python${PYTHON_VERSION}-${OS}-${IMAGE_TAG_SUFFIX}"
	export IMAGE="$image"

	for file in $(find "$TEST_PATH" -type f -iname "[0-9]*-*.bats" | sort); do
		bats --tap "$file"
	done
}

main "$@"
