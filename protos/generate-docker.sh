#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

echo "Building BentoML proto generator docker image.."
# Avoid using source code directory as docker build context to have a faster build
docker build -t bentoml-proto-generator - <<EOF
FROM python:3.7

RUN apt-get update && apt-get install -y nodejs npm

RUN npm install -g protobufjs@6.7.0

# let pbjs install all additional protobuf.js CLI depdendencies
RUN pbjs --help &> /dev/null

RUN pip install grpcio-tools==1.27.2
EOF

echo "Starting BentoML proto generator docker container.."
docker run --rm -v $GIT_ROOT:/home/bento bentoml-proto-generator \
    bash -c "cd /home/bento/ && ./protos/generate.sh"
