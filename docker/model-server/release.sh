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

PYTHON_MAJOR_VERSIONS=(3.6 3.7 3.8)
echo "Building slim docker base images for ${PYTHON_MAJOR_VERSIONS[*]}"
for version in "${PYTHON_MAJOR_VERSIONS[@]}"
do
    echo "Releasing slim docker base image for Python $version.."
    docker build --pull \
    --build-arg BENTOML_VERSION=$BENTOML_VERSION \
    --build-arg PYTHON_VERSION=$version \
    -t bentoml/model-server:$BENTOML_VERSION-slim-py${version//.} \
    -f Dockerfile-slim \
    --network=host \
    .

    docker push bentoml/model-server:$BENTOML_VERSION-slim-py${version//.}

done
echo "Done"
