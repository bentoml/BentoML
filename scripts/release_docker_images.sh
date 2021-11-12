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

manager_dockerfiles="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml-docker python3 manager.py"

manager_images="docker run --rm -v $(pwd):/bentoml -v /var/run/docker.sock:/var/run/docker.sock bentoml-docker python3 manager.py"

cd "$DOCKER_DIR" || exit

log "Creating new Manager container if one doesn't exist."
if [[ $(docker images --filter=reference='bentoml-docker' -q) == "" ]] | [[ $(git diff "$DOCKER_DIR"/Dockerfile) != "" ]]; then
		DOCKER_BUILDKIT=1 docker build -t bentoml-docker -f Dockerfile .
fi

# We will always run this scripts after release_pypi.sh, so the branch of new releases will exist.
if git rev-parse "$VERSION_STR" >/dev/null 2>&1; then
  # Tags already exists
  git checkout "$VERSION_STR"
fi

log "Generating new Dockerfiles for BentoML $VERSION_STR..."
$manager_dockerfiles --bentoml_version "$BENTOML_VERSION" --generate dockerfiles

if [[ ! -f "$DOCKER_DIR"/.env ]];
  warn "Make sure to create a $DOCKER_DIR/.env to setup docker registry correctly. Refers to manifest.yml to see which envars you need to setup."
  exit 1
fi

log "Building docker image for BentoML $VERSION_STR and perform push to registries..."
$manager_images --bentoml_version "$BENTOML_VERSION" --generate images --push_to_hub