#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
yapf -i $(find $GIT_ROOT/bentoml -name "*.py")