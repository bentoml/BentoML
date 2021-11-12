#!/usr/bin/env bash

source ./scripts/ci/helpers.sh

echo "Running black format check..."

set_on_failed_callback "FAIL black errors"

if ! (black --check --config "./pyproject.toml" bentoml tests docker); then
  FAIL "black format check failed"
  echo "Make sure to run \`make format\`"
  exit 1
fi

PASS "black format check passed!"
exit 0