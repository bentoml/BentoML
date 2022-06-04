#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config "$GIT_ROOT/pyproject.toml" bentoml tests docs

INFO "(isort) Reordering imports..."

isort "$GIT_ROOT"

INFO "(setup-cfg-fmt) Format setup.cfg ..."

setup-cfg-fmt "$GIT_ROOT/setup.cfg" --min-py3-version 3.7 --max-py-version 3.10
