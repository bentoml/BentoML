#!/usr/bin/env bash

source ./scripts/ci/helpers.sh

set_on_failed_callback "FAIL pyright errors"

echo "Running pyright..."
if [[ -n $GITHUB_BASE_REF ]]; then
  echo "Running pyright on changed files..."
  git fetch origin "$GITHUB_BASE_REF"
  if ! (git diff --name-only --diff-filter=d "origin/$GITHUB_BASE_REF" -z -- '*.py' | xargs -0 --no-run-if-empty pyright); then
    FAIL "pyright failed."
    exit 1
  fi
elif ! (pyright .); then
  FAIL "pyright failed."
  exit 1
fi

PASS "pyright passed!"
exit 0
