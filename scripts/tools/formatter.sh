#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config ./pyproject.toml bentoml/ tests/ docker/

INFO "(black) Formatting VCS stubs..."

git ls-files -z -cm -- typings ':!:*.md' | xargs -0 -I {} -i sh -c 'echo "Processing "{}"..."; black --config ./pyproject.toml --pyi {}'

INFO "(isort) Reordering imports..."

isort .
