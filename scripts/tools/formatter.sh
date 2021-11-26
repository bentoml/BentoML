#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

black --config ./pyproject.toml bentoml/ tests/ docker/ typings/

isort bentoml/ tests/ docker/ typings/
