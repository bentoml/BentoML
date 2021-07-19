#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

echo "Running flake8 on bentoml, yatai module..."
flake8 --config=.flake8 bentoml yatai/yatai

echo "Running flake8 on docker directory..."
flake8 --config=.flake8 docker/manager.py docker/utils.py

echo "Running flake8 on test module..."
flake8 --config=.flake8 tests

echo "Running pylint on bentoml, yatai module..."
pylint --rcfile="./pylintrc" bentoml yatai/yatai

echo "Running pylint on docker directory..."
pylint --rcfile="./pylintrc" docker/manager.py docker/utils.py

echo "Running pylint on test module..."
pylint --rcfile="./pylintrc" tests

echo "Running mypy on bentoml, yatai module..."
mypy --config=mypy.ini --show-error-codes --no-incremental bentoml yatai

echo "Running mypy on docker module..."
mypy --config=mypy.ini --show-error-codes --no-incremental docker

echo "Done"