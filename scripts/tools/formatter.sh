#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config ./pyproject.toml src tests docs examples

INFO "(black) Formatting stubs..."

find src -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config ./pyproject.toml {} \;

INFO "(isort) Reordering imports..."

isort .
