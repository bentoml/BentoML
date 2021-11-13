#!/usr/bin/env bash

name=$(basename "$0")

source ./scripts/ci/helpers.sh

set_on_failed_callback "FAIL pyright errors"

echo "Running pyright..."
if ! (pyright .); then
  FAIL "pyright failed."
  exit 1
fi

PASS "pyright passed!"
exit 0
