#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config ./pyproject.toml bentoml/ tests/ docker/

INFO "(black) Formatting stubs..."

black --pyi typings/**/*.pyi

INFO "(isort) Reordering imports..."

isort .
