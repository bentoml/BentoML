#!/usr/bin/env bash

set -e

TEST_DIR="$(cd "$( dirname "$(readlink -f $0)")" && pwd)"

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
case $1 in
    -img|--image_name)
    IMAGE_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    -bv|--bentoml_version)
    BENTOML_VERSION="$2"
    shift # past argument
    shift # past value
    ;;
    -pv|--python_version)
    PYTHON_VERSION="$2"
    shift # past argument
    shift # past value
    ;;
    -os|--distros)
    OS="$2"
    shift # past argument
    shift # past value
    ;;
    -sf|--suffix)
    IMAGE_TAG_SUFFIX="$2"
    shift # past argument
    shift # past value
    ;;
    -pl|--platforms)
    PLATFORMS="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    BENTOML_VERSION=$(git describe --tags $(git rev-list --tags --max-count=1) | sed 's/v\(\)/\1/; s/-//g')
    IMAGE_NAME='bento-server'
    shift # past argument
    ;;
    -*|--*)
    echo "Unknown option $1"
    exit 1
    ;;
    *)
    POSITIONAL_ARGS+=("$1") # save positional arg
    shift # past argument
    ;;
esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

main(){
    # TODO: add debian11 and debian10 with new releases
    export __TEST_PYTHON_VERSION="$PYTHON_VERSION"
    export __TEST_OS="$OS"
    export __TEST_IMAGE_TAG_SUFFIX="$IMAGE_TAG_SUFFIX"
    export __TEST_IMAGE_NAME="$IMAGE_NAME"
    export __TEST_BENTOML_VERSION="$BENTOML_VERSION"
    export __TEST_IMAGE_NAME="$IMAGE_NAME"
    export __TEST_ARCH="$PLATFORMS"
    # export __TEST_ARCH=$(uname -m) when we have multi-arch

    . $TEST_DIR/scripts/run_tests
}

main "$@"
