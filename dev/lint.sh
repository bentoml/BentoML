#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

log(){
	echo -e "\033[2mINFO::\033[0m \e[1;32m$*\e[m" 1>&2
}

log "Running flake8 on bentoml, yatai directory..."
flake8 --config=.flake8 bentoml yatai/yatai

log "Running flake8 on test, docker directory..."
flake8 --config=.flake8 tests docker

log "Running pylint on bentoml, yatai directory..."
pylint --rcfile="./pylintrc" bentoml yatai/yatai

log "Running pylint on test, docker directory..."
pylint --rcfile="./pylintrc" tests docker

log "Running mypy on bentoml, yatai directory..."
mypy --config-file "$GIT_ROOT"/mypy.ini bentoml yatai

log "Running mypy on docker directory..."
mypy --config-file "$GIT_ROOT"/mypy.ini docker

log "Done"
