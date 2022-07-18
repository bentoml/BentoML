#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config ./pyproject.toml bentoml tests docs examples

INFO "(isort) Reordering imports..."

isort .

INFO "(buf) Formatting protobuf..."

ARGS=(--config "/workspace/protos/buf.yaml" format -w /workspace/protos/)
if which buf &>/dev/null; then
  buf "${ARGS[@]}"
else
  docker run --rm --volume "$(pwd)":/workspace --workdir /workspace bufbuild/buf "${ARGS[@]}"
fi
