#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)


gen_protos_docker(){
  echo "Building BentoML proto generator docker image.."
  # Avoid using source code directory as docker build context to have a faster build
  docker build -t bentoml-proto-generator - <<EOF
FROM python:3.7

RUN pip install grpcio-tools~=1.34.0 mypy-protobuf

RUN apt-get update && apt-get install -y nodejs npm

RUN npm install -g npm@latest

RUN npm install -g protobufjs@6.9.0
EOF
}

if [[ $(docker images --filter=reference='bentoml-proto-generator' -q) == "" ]]; then
  gen_protos_docker
fi

echo "Starting BentoML proto generator docker container.."
docker run --rm -v "$GIT_ROOT":/home/bentoml bentoml-proto-generator \
        bash -c "BENTOML_REPO=/home/bentoml /home/bentoml/protos/generate.sh"