#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
sphinx-build -b html $GIT_ROOT/docs $GIT_ROOT/built-docs
