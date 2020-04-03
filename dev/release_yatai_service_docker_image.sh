#!/bin/bash
set -e

if [ "$#" -eq 1 ]; then
  BENTOML_VERSION=$1
else
  echo "Must provide target BentoML version, e.g. ./script/release_yatai_service_docker_image.sh 0.7.0"
  exit 1
fi


GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT/docker/yatai-service

docker build --pull \
    --build-arg version=$BENTOML_VERSION \
    -t bentoml/yatai-service:$BENTOML_VERSION \
    .

docker push bentoml/yatai-service:$BENTOML_VERSION