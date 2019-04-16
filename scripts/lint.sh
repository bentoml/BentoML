#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT

pylint --rcfile="./pylintrc" bentoml
pylint --rcfile="./pylintrc" tests
pycodestyle bentoml
pycodestyle tests
