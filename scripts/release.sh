#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd $GIT_ROOT

# Currently only BentoML maintainer has permission to create new pypi
# releases
if [ ! -f $HOME/.pypirc ]; then
  # about .pypirc file:
  # https://docs.python.org/3/distutils/packageindex.html#the-pypirc-file
  echo "Error: File \$HOME/.pypirc not found."
  exit
fi

echo "Installing dev dependencies..."
pip install .[dev]

echo "Generating distribution archives..."
python3 setup.py sdist bdist_wheel

# Use testpypi by default, run script with: "REPO=pypi release.sh" for
# releasing to Pypi.org
REPO=${REPO:=testpypi}

echo "Uploading package to $REPO..."
twine upload --repository $REPO dist/*
