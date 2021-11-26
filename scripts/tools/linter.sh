#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit

flake8 --config=setup.cfg bentoml tests docker

pylint --rcfile="./pyproject.toml" bentoml

pylint --rcfile="./pyproject.toml" --disable=E0401,F0010 tests docker
