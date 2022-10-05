#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config ./pyproject.toml bentoml bentoml_cli tests docs examples

INFO "(black) Formatting stubs..."

find bentoml -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config ./pyproject.toml {} \;

INFO "(isort) Reordering imports..."

isort .
