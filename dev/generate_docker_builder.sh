#!/usr/bin/env bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT"/docker || exit

if [[ $(docker images --filter=reference='bentoml-docker' -q) == "" ]] | [[ $(git diff Dockerfile) != "" ]]; then \
		DOCKER_BUILDKIT=1 docker build -t bentoml-docker -f Dockerfile .; \
fi;

echo -e "\e[0;33m"
cat <<EOF
Remember to create both alias

manager_dockerfiles='docker run --rm -u \$(id -u):\$(id -g) -v \$(pwd):/bentoml bentoml-docker python3 manager.py '
and

manager_images='docker run --rm -v \$(pwd):/bentoml -v /var/run/docker.sock:/var/run/docker.sock bentoml-docker python3 manager.py '

in order to use docker container correctly
EOF
echo -e "\e[m"