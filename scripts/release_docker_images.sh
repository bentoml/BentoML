#!/usr/bin/env bash
set -ex
shopt -s expand_aliases

if [ "$#" -eq 1 ]; then
  BENTOML_VERSION=$1
else
  echo "Must provide target BentoML version, e.g. ./dev/release_docker_images.sh 0.12.1"
  exit 1
fi

log(){
	echo -e "\033[2mINFO::\033[0m \e[1;32m$*\e[m" 1>&2
}

warn() {
	echo -e "\033[2mWARN::\033[0m \e[1;33m$*\e[m" 1>&2
}

GIT_ROOT=$(git rev-parse --show-toplevel)
VERSION_STR="v$BENTOML_VERSION"
DOCKER_DIR="$GIT_ROOT"/docker

cd "$DOCKER_DIR" || exit


log "Creating new Manager container if one doesn't exist."
if [[ $(docker images --filter=reference='bentoml-docker' -q) == "" ]]; then
	make docker-bake-manager
fi

log "Check out a new branch for releases $VERSION_STR"
git checkout -b releases/docker/${VERSION_STR}

log "Generating new Dockerfiles for BentoML $VERSION_STR..."
make docker-run-generate


if [[ ! -f "$DOCKER_DIR"/.env ]]; then
  warn "Make sure to create a $DOCKER_DIR/.env to setup docker registry correctly. Refers to ${DOCKER_DIR}/.env.example to see which envars you need to setup."
  exit 1
fi

log "Login into DockerHub..."
make docker-run-authenticate

log "Building docker image for BentoML $VERSION_STR and perform push to registries..."
make docker-run-build max-workers=5 registry=docker.io

