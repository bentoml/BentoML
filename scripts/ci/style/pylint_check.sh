#!/usr/bin/env bash

name=$(basename "$0")

source ./scripts/ci/helpers.sh

GIT_ROOT=$(git rev-parse --show-toplevel)

set_on_failed_callback "FAIL pylint errors"

if [[ -n "$GITHUB_BASE_REF" ]]; then
  echo "Running pylint on changed files..."
  git fetch origin "$GITHUB_BASE_REF"
  if ! (git diff --name-only --diff-filter=d "origin/$GITHUB_BASE_REF" -z -- '*.py' | xargs -0 --no-run-if-empty pylint --rcfile="$GIT_ROOT/pyproject.toml"); then
    FAIL "pylint failed."
    exit 1
  fi
fi

PASS "pylint check passed!"
exit 0
