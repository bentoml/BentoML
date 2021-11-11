#!/usr/bin/env bash
set -x

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

has_errors=0

# Code auto formatting check with black & isort
./dev/format.sh
GIT_STATUS="$(git status --porcelain)"
if [ "$GIT_STATUS" ];
then
  echo "Source code changes are not formatted (./dev/format.sh script)"
  echo "Files changed:"
  echo "--------------------------------------------------------------"
  echo "$GIT_STATUS"
  has_errors=1
else
  echo "Code auto formatting passed"
fi

# The first line of the tests are always empty if there are no linting errors

echo "Running flake8 on bentoml module.."
output=$( flake8 --config=setup.cfg bentoml )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

echo "Running flake8 on tests and docker module.."
output=$( flake8 --config=setup.cfg tests docker )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

# echo "Running pylint on bentoml module.."
# output=$( pylint --rcfile="./pylintrc" bentoml )
# first_line=$(echo "${output}" | head -1)
# echo "$output"
# if [ -n "$first_line" ]; then
#   has_errors=1
# fi

echo "Running pylint on tests and docker module.."
output=$( pylint --rcfile="./pylintrc" --disable=E0401,F0010 tests docker )
first_line=$(echo "${output}" | head -1)
echo "$output"
if [ -n "$first_line" ]; then
  has_errors=1
fi

# echo "Running mypy on bentoml module.."
# output=$( mypy --config-file "$GIT_ROOT"/mypy.ini bentoml )
# first_line=$(echo "${output}" | head -1)
# echo "$output"
# if [ -n "$first_line" ]; then
#   # has_errors=1
# fi

# echo "Running mypy on docker module.."
# output=$( mypy --config-file "$GIT_ROOT"/mypy.ini docker )
# first_line=$(echo "${output}" | head -1)
# echo "$output"
# if [ -n "$first_line" ]; then
#   # has_errors=1
# fi

if [[ -n $GITHUB_BASE_REF ]]; then
  echo "Running pyright on changed files..."
  git fetch origin "$GITHUB_BASE_REF"
  if ! (git diff --name-only --diff-filter=d "origin/$GITHUB_BASE_REF" -z -- '*.py' | xargs -0 --no-run-if-empty pyright); then
    has_errors=1
  fi
fi

echo "Done"
exit $has_errors
