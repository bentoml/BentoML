#!/usr/bin/env bash

GIT_ROOT="$(git rev-parse --show-toplevel)"

cd "$GIT_ROOT" || exit 1

log(){
	echo -e "\033[2mINFO::\033[0m \e[1;32m$*\e[m" 1>&2
}

log "Running black and isort on bentoml directory..."
black --config "$GIT_ROOT"/pyproject.toml bentoml/
isort bentoml

log "Running black and isort on tests and docker directory..."
black --config "$GIT_ROOT"/pyproject.toml tests/ docker/
isort tests/ docker/
