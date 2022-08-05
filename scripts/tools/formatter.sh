#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(black) Formatting codebase..."

black --config ./pyproject.toml bentoml bentoml_cli tests docs examples

INFO "(isort) Reordering imports..."

isort .

INFO "(buf) Formatting protobuf..."

docker run --rm --volume "$GIT_ROOT":/workspace --workdir /workspace bufbuild/buf format --config "/workspace/bentoml/grpc/buf.yaml" -w "/workspace/bentoml/"
