#!/usr/bin/env bash
# Bash Script that installs the dependencies specified in the BentoService archive
set -ex

# cd to the saved bundle directory
SAVED_BUNDLE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd $SAVED_BUNDLE_PATH

# Run the user defined setup.sh script if it is presented
if [ -f ./setup.sh ]; then chmod +x ./setup.sh && bash -c ./setup.sh; fi

# Install conda dependencies to base env if conda command is available
command -v conda >/dev/null && conda env update -n base -f ./environment.yml \
  || echo "conda command not found, ignoring environment.yml"

# Install PyPI packages specified in requirements.txt
pip install -r ./requirements.txt --no-cache-dir

# install sdist or wheel format archives stored under bundled_pip_dependencies directory
for filename in ./bundled_pip_dependencies/*.tar.gz; do
  [ -e "$filename" ] || continue
  pip install -U "$filename"
done
