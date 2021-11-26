#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

source "$GIT_ROOT/scripts/ci/helpers.sh"

INFO "Formatting related files..."
black --config ./pyproject.toml bentoml tests docker
isort --color bentoml tests docker

INFO "Formatting stubs files..."
black --pyi typings/**/*.pyi
