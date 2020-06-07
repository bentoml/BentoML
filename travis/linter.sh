#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT

# Code auto formatting check with black
pip install "black>=19.0"
black -S .
GIT_STATUS="$(git status --porcelain)"
if [ "$GIT_STATUS" ];
then
  echo "Source code changes are not formated with black(./dev/format.sh script)"
  echo "Files changed:"
  echo "------------------------------------------------------------------"
  echo "$GIT_STATUS"
  has_errors=1
else
  echo "Code auto formatting passed"
fi

# The first line of the tests are  
# always empty if there are no linting errors

has_errors=0

echo "Running flake8 on bentoml module.."
output=$( flake8 --config=.flake8 bentoml )
firstline=`echo "${output}" | head -1`
echo "$output"
if ! [ -z "$firstline" ]; then
  has_errors=1
fi

echo "Running flake8 on test module.."
output=$( flake8 --config=.flake8 tests e2e_tests )
firstline=`echo "${output}" | head -1`
echo "$output"
if ! [ -z "$firstline" ]; then
  has_errors=1
fi

echo "Running pylint on bentoml module.."
output=$( pylint --rcfile="./pylintrc" bentoml )
firstline=`echo "${output}" | head -1`
echo "$output"
if ! [ -z "$firstline" ]; then
  has_errors=1
fi

echo "Running pylint on test module.."
output=$( pylint --rcfile="./pylintrc" tests e2e_tests )
firstline=`echo "${output}" | head -1`
echo "$output"
if ! [ -z "$firstline" ]; then
  has_errors=1
fi

echo "Done"
exit $has_errors
