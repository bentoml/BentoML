#!/usr/bin/env bash

name=$(basename "$0")

source ./scripts/ci/helpers.sh

echo "Running pylint..."

set_on_failed_callback "FAIL pylint errors"

pylint --rcfile="./pyproject.toml" bentoml | tee /tmp/"$name"_bentoml
pylint --rcfile="./pyproject.toml" --disable=E0401,F0010 tests docker | tee /tmp/"$name"_tests_docker

if [ -s /tmp/"$name"_tests_docker ] || [ -s /tmp/"$name"_bentoml ]; then
  FAIL "pylint failed"
  cat /tmp/"$name"_tests_docker || cat /tmp/"$name"_bentoml
  exit 1
fi

PASS "pylint check passed!"
exit 0
