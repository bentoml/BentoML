#!/usr/bin/env bash
set -e

if ! [ -x "$(command -v docker)" ]; then
  echo "Please install docker first, or run the generate script directly"
  exit 1
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

echo "Start BentoML proto-dev docker container"
echo $GIT_ROOT

docker run --rm -v $GIT_ROOT:/home/bento --name generate-proto bentoml/proto-dev /home/bento/protos/generate.sh

echo "Generate python and javascript files from protobuf complete."

