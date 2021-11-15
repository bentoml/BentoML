#!/usr/bin/env bash

source ./scripts/ci/helpers.sh

echo "Running isort format check..."

set_on_failed_callback "FAIL isort errors"

if ! (isort --check bentoml tests docker typings); then
  FAIL "isort format check failed"
  echo "Make sure to run \`make format\`"
  exit 1
fi

PASS "isort format check passed!"
exit 0
