#!/usr/bin/env bash
set -ex

GIT_ROOT=$(git rev-parse --show-toplevel)


gen_protos_docker(){
  echo "Building BentoML proto generator docker image.."
  # Avoid using source code directory as docker build context to have a faster build
  docker build -t bentoml-proto-generator - <<<"$(cat "$GIT_ROOT"/protos/Dockerfile)"
}

if [[ $(docker images --filter=reference='bentoml-proto-generator' -q) == "" ]]; then
  gen_protos_docker
fi

echo "Starting BentoML proto generator docker container.."
docker run --rm -u "$(id -u)":"$(id -g)" -v "$GIT_ROOT":/home/bentoml/workspace bentoml-proto-generator \
      bash -c "BENTOML_REPO=/home/bentoml/workspace . \$NVM_DIR/nvm.sh && . protos/generate.sh"