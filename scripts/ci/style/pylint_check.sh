#!/usr/bin/env bash

name=$(basename "$0")

source ./scripts/ci/helpers.sh

GIT_ROOT=$(git rev-parse --show-toplevel)

set_on_failed_callback "FAIL pylint errors"

if [[ -n "$GITHUB_BASE_REF" ]]; then
  echo "Running pylint on changed files..."
  git fetch origin "$GITHUB_BASE_REF"
  if ! (git diff --name-only --diff-filter=d "origin/$GITHUB_BASE_REF" -z -- '*.py' | xargs -0 --no-run-if-empty pylint --rcfile="$GIT_ROOT/pyproject.toml" --exit-zero); then
    FAIL "pylint failed."
    exit 1
  fi
else
  echo "Running pylint for the whole library..."
  pylint --rcfile="$GIT_ROOT/pyproject.toml" bentoml | tee /tmp/"$name"_bentoml
  pylint --rcfile="$GIT_ROOT/pyproject.toml" --disable=E0401,F0010 tests docker | tee /tmp/"$name"_tests_docker

  if [ -s /tmp/"$name"_tests_docker ] || [ -s /tmp/"$name"_bentoml ]; then
    FAIL "pylint failed"
    cat /tmp/"$name"_tests_docker || cat /tmp/"$name"_bentoml
    exit 1
  fi
fi

PASS "pylint check passed!"
exit 0
