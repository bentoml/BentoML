#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

INFO "(pylint) Linting bentoml..."

pylint --rcfile="./pyproject.toml" --fail-under 9.5 bentoml bentoml_cli

INFO "(pylint) Linting examples..."

pylint --rcfile="./pyproject.toml" --fail-under 9.0 --disable=W0621,E0611 examples

INFO "(pylint) Linting tests..."

pylint --rcfile="./pyproject.toml" --disable=E0401,F0010 tests

INFO "(yamllint) Linting yaml files..."

find "$GIT_ROOT" -type f -iname "*.yml" -exec yamllint -c "$GIT_ROOT/.yamllint.yml" {} \;

INFO "(buf) Linting protobuf..."

docker run --rm --volume "$GIT_ROOT":/workspace --workdir /workspace bufbuild/buf lint --config "/workspace/bentoml/grpc/buf.yaml" --error-format msvs
