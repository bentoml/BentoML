#!/usr/bin/env bash
set -x

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

has_errors=0

# Code auto formatting check with black & isort
echo "Running black on bentoml module..."
if ! (black --check --config "$GIT_ROOT/pyproject.toml" bentoml); then
  has_errors=1
fi

echo "Running black on tests and docker modules..."
if ! (black --check --config "$GIT_ROOT/pyproject.toml" tests docker); then
  has_errors=1
fi

echo "Running isort on bentoml module..."
if ! (isort --check bentoml); then
  has_errors=1
fi

echo "Running isort on tests and docker modules..."
if ! (isort --check tests docker); then
  has_errors=1
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
