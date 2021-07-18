#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

echo "Running flake8 on bentoml, docker, and yatai module..."
flake8 --config=.flake8 bentoml docker yatai

echo "Running flake8 on test module..."
flake8 --config=.flake8 tests

echo "Running pylint on bentoml, docker, and yatai module..."
pylint --rcfile="./pylintrc" bentoml docker yatai

echo "Running pylint on test module..."
pylint --rcfile="./pylintrc" tests

echo "Running mypy on bentoml module..."
mypy --config=mypy.ini --show-error-codes --no-incremental bentoml

echo "Running mypy on yatai module..."
mypy --config=mypy.ini --show-error-codes --no-incremental yatai

echo "Done"