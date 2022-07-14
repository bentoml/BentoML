#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

set_on_failed_callback "[FAIL] proto check failed"

echo "Running proto format check..."

if ! (git diff --name-only --diff-filter=d "origin/$GITHUB_BASE_REF" -z -- '*.proto' | xargs -0 --no-run-if-empty buf format --config "./protos/buf.yaml" -d --exit-code); then
  FAIL "proto format check failed"
  echo "Format incorrectly. Make sure to run \`make format\`"
  exit 1
fi

PASS "proto format check passed!"

echo "Running proto lint check..."

if ! (git diff --name-only --diff-filter=d "origin/$GITHUB_BASE_REF" -z -- '*.proto' | xargs -0 --no-run-if-empty buf lint --config "./protos/buf.yaml" --error-format msvs); then
  FAIL "proto lint check failed"
  echo "Lint error. Make sure to run \`make lint\`"
  exit 1
fi

PASS "proto lint check passed!"

PASS "proto check passed!"
exit 0
