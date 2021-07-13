#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

echo "Running flake8 on bentoml module.."
flake8 --config=.flake8 bentoml docker

echo "Running flake8 on test module.."
flake8 --config=.flake8 tests e2e_tests

echo "Running pylint on bentoml module.."
pylint --rcfile="./pylintrc" bentoml

echo "Running pylint on test module.."
pylint --rcfile="./pylintrc" tests e2e_tests

echo "Done"