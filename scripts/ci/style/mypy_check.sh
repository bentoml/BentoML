#!/usr/bin/env bash

name=$(basename "$0")

source ./scripts/ci/helpers.sh

echo "Running mypy..."

set_on_failed_callback "FAIL mypy errors"

mypy --config-file ./pyproject.toml bentoml/**/*.py | tee /tmp/"$name"_bentoml
mypy --config-file ./pyproject.toml docker/**/*.py | tee /tmp/"$name"_docker

if [ -s /tmp/"$name"_docker ] || [ -s /tmp/"$name"_bentoml ]; then
  FAIL "mypy failed"
  cat /tmp/"$name"_docker || cat /tmp/"$name"_bentoml
  exit 1
fi

PASS "mypy check passed!"
exit 0
