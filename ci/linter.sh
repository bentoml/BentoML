#!/usr/bin/env bash
set -x

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

has_errors=0

# Code auto formatting check with black
black -S .
GIT_STATUS="$(git status --porcelain)"
if [ "$GIT_STATUS" ];
then
  echo "Source code changes are not formatted with black (./dev/format.sh script)"
  echo "Files changed:"
  echo "------------------------------------------------------------------"
  echo "$GIT_STATUS"
  has_errors=1
else
  echo "Code auto formatting passed"
fi

# The first line of the tests are  
# always empty if there are no linting errors

echo "Running flake8 on bentoml module.."
output=$( flake8 --config=.flake8 bentoml )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

echo "Running flake8 on test module.."
output=$( flake8 --config=.flake8 tests e2e_tests )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

echo "Running pylint on bentoml module.."
output=$( pylint --rcfile="./pylintrc" bentoml )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

echo "Running pylint on test module.."
output=$( pylint --rcfile="./pylintrc" tests e2e_tests )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

echo "Done"
exit $has_errors
