#!/usr/bin/env bash
set -e

if [ "$#" -eq 1 ]; then
  export BENTOML_VERSION=$1
else
  echo "Must provide target BentoML version, e.g. ./script/release_yatai_service_docker_image.sh 0.7.0"
  exit 1
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

./docker/yatai-service/release.sh
./docker/model-server/release.sh
