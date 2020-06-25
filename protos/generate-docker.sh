#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

if ! [ -x "$(command -v docker)" ]; then
  echo "Please install docker first, or run the generate script directly"
  exit 1
fi

if [[ "$(docker images -q bentoml/proto-dev 2> /dev/null)" == "" ]]; then
  echo "Docker image bentoml/protp-dev not found, buidling docker image now"
  docker build -t bentoml/proto-dev .
fi

echo "Start BentoML proto-dev docker container"

docker run --rm -v $GIT_ROOT:/home/bento --name generate-proto bentoml/proto-dev bash -c "cd /home/bento/ && ./protos/generate.sh"

echo "Generate python and javascript files from protobuf complete."

