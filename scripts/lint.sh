#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT

echo "Running pylint on bentoml module.."
pylint --rcfile="./pylintrc" bentoml

echo "Running pylint on test module.."
pylint --rcfile="./pylintrc" tests

echo "Running pycodestyle on bentoml module.."
pycodestyle bentoml

echo "Running pycodestyle on test module.."
pycodestyle tests

echo "Done"
