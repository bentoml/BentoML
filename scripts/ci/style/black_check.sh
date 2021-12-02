#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

echo "Running black format check..."

set_on_failed_callback "[FAIL] black format check failed"

if ! (black --check --config "./pyproject.toml" bentoml tests docker typings); then
  FAIL "black format check failed"
  echo "Make sure to run \`make format\`"
  exit 1
fi

PASS "black format check passed!"
exit 0
