#!/usr/bin/env bash

name=$(basename "$0")

source ./scripts/ci/helpers.sh

echo "Running flake8 lint..."

set_on_failed_callback "FAIL flake8 errors"

flake8 --config=./setup.cfg bentoml tests docker | tee /tmp/"$name"

if [ -s /tmp/"$name" ]; then
  FAIL "flake8 failed"
  cat /tmp/"$name"
  exit 1
fi

PASS "flake8 check passed!"
exit 0
