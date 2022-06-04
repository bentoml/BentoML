#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

source ./scripts/ci/helpers.sh

echo "Running isort format check..."

set_on_failed_callback "[FAIL] isort format check failed"

if ! (isort --check "$GIT_ROOT"); then
	FAIL "isort format check failed"
	echo "Make sure to run \`make format\`"
	exit 1
fi

PASS "isort format check passed!"
exit 0
