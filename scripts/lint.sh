#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT

pylint --rcfile="./pylintrc" bentoml
pycodestyle bentoml
pycodestyle tests
