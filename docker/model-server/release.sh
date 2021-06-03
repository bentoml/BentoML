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

PYTHON_MAJOR_VERSIONS=(3.6 3.7 3.8)
#which version to tag with latest tag
PYTHON_LATEST_VERSION=3.7

echo "Building debian based docker base images for ${PYTHON_MAJOR_VERSIONS[*]}"
for version in "${PYTHON_MAJOR_VERSIONS[@]}"
do
    echo "Releasing debian based docker base image for Python $version .."
    docker build --pull \
        --build-arg BENTOML_VERSION="$BENTOML_VERSION" \
        --build-arg PYTHON_VERSION="$version" \
        -t bentoml/model-server:"$BENTOML_VERSION"-py"${version//.}" \
        -t bentoml/model-server:latest-py"${version//.}" \
        .
    if ! [[ -t 1 ]]; then  # differentiate when developing Dockerfile and running ./dev/release_docker_images.sh
      docker push bentoml/model-server:"$BENTOML_VERSION"-py"${version//.}"
      docker push bentoml/model-server:latest-py"${version//.}"
    fi

done

# tag the default version as both latest and unspecified python version
docker tag bentoml/model-server:latest-py${PYTHON_LATEST_VERSION//.} bentoml/model-server:latest
docker tag bentoml/model-server:"$BENTOML_VERSION"-py${PYTHON_LATEST_VERSION//.} bentoml/model-server:"$BENTOML_VERSION"

if ! [[ -t 1 ]]; then
  docker push bentoml/model-server:latest
  docker push bentoml/model-server:"$BENTOML_VERSION"
fi

echo "Building slim docker base images for ${PYTHON_MAJOR_VERSIONS[*]}"
for version in "${PYTHON_MAJOR_VERSIONS[@]}"
do
    echo "Releasing slim docker base image for Python $version.."
    docker build --pull \
    --build-arg BENTOML_VERSION="$BENTOML_VERSION" \
    --build-arg PYTHON_VERSION="$version" \
    -t bentoml/model-server:"$BENTOML_VERSION"-slim-py"${version//.}" \
    -t bentoml/model-server:latest-slim-py"${version//.}" \
    -f Dockerfile-slim \
    --network=host \
    .

    if ! [[ -t 1 ]]; then
      docker push bentoml/model-server:"$BENTOML_VERSION"-slim-py"${version//.}"
      docker push bentoml/model-server:latest-slim-py"${version//.}"
    fi

done

echo "Building cuda-enabled docker base images for ${PYTHON_MAJOR_VERSIONS[*]}"
for version in "${PYTHON_MAJOR_VERSIONS[@]}"
do
    echo "Releasing cuda-enabled docker base image for Python $version.."
    # for PYTHON_VERSION we are using bentoml/model-server:slim images to reduce boilerplate and decrease buildtime
    docker build --pull \
    --build-arg BENTOML_VERSION="$BENTOML_VERSION" \
    --build-arg PYTHON_VERSION="${version//.}" \
    -t bentoml/model-server:"$BENTOML_VERSION"-py"${version//.}"-gpu \
    -t bentoml/model-server:latest-py"${version//.}"-gpu \
    -f Dockerfile-gpu \
    .

    if ! [[ -t 1 ]]; then
      docker push bentoml/model-server:"$BENTOML_VERSION"-py"${version//.}"-gpu
      docker push bentoml/model-server:latest-py"${version//.}"-gpu
    fi
done
echo "Done"