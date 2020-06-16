#!/usr/bin/env bash
set -e

if [ "$#" -eq 1 ]; then
  BENTOML_VERSION=$1
else
  echo "Must provide target BentoML version, e.g. ./release.sh 0.7.0"
  exit 1
fi


GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"/docker/model-server

echo "Releasing debian based docker base image.."
docker build --pull \
    --build-arg BENTOML_VERSION="$BENTOML_VERSION" \
    -t bentoml/model-server:"$BENTOML_VERSION" \
    .
docker push bentoml/model-server:"$BENTOML_VERSION"

echo "Releasing alpine based docker base image.."
docker build --pull \
    --build-arg BENTOML_VERSION="$BENTOML_VERSION" \
    -t bentoml/model-server:"$BENTOML_VERSION"-alpine \
    -f Dockerfile-alpine \
    --network=host \
    .
docker push bentoml/model-server:"$BENTOML_VERSION"-alpine

echo "Done"
