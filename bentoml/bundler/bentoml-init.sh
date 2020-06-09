#!/usr/bin/env bash
set -ex

SAVED_BUNDLE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd $SAVED_BUNDLE_PATH

# run user defined setup script
if [ -f ./setup.sh ]; then chmod +x ./setup.sh && bash -c ./setup.sh; fi

# update conda base env if conda command is available in base image
command -v conda >/dev/null && conda env update -n base -f ./environment.yml \
  || echo "conda command not found, ignoring environment.yml"

# install pip dependencies
pip install -r ./requirements.txt

# install bundled_pip_dependencies
for filename in ./bundled_pip_dependencies/*.tar.gz; do
  [ -e "$filename" ] || continue
  pip install -U "$filename"
done