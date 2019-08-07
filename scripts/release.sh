#!/bin/bash
set -e

if [ "$#" -eq 1 ]; then
  VERSION_STR=$1
else
  echo "Must provide release version string, e.g. ./script/release.sh 1.0.5"
  exit 0
fi

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


if [ -d $GIT_ROOT/dist ]; then
  echo "Removing existing 'dist' directory"
  rm -rf $GIT_ROOT/dist
fi

tag_name="bentoml-release-v$VERSION_STR"
git tag -a $tag_name -m "Tag generated with BentoML/script/release.sh, version: $VERSION_STR"

echo "Installing dev dependencies..."
pip install .[dev]

echo "Generating distribution archives..."
python3 setup.py sdist bdist_wheel

# Use testpypi by default, run script with: "REPO=pypi release.sh" for
# releasing to Pypi.org
REPO=${REPO:=testpypi}

echo "Uploading package to $REPO..."
twine upload --repository $REPO dist/* --verbose

git push origin $tag_name
