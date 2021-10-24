#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

log(){
	echo -e "\033[2mINFO::\033[0m \e[1;32m$*\e[m" 1>&2
}

log "Running flake8 on bentoml directory..."
flake8 --config=setup.cfg bentoml

log "Running flake8 on test, docker directory..."
flake8 --config=setup.cfg tests docker

log "Running pylint on bentoml directory..."
pylint --rcfile="./pylintrc" bentoml

log "Running pylint on test, docker directory..."
pylint --rcfile="./pylintrc" --disable=E0401 tests docker

log "Running mypy on bentoml directory..."
mypy --config-file "$GIT_ROOT"/mypy.ini bentoml

log "Running mypy on docker directory..."
mypy --config-file "$GIT_ROOT"/mypy.ini docker

log "Done"
