#!/usr/bin/env bash
# Bash Script that installs the dependencies specified in the BentoService archive
set -ex

# cd to the saved bundle directory
SAVED_BUNDLE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd $SAVED_BUNDLE_PATH

# Run the user defined setup.sh script if it is presented
if [ -f ./setup.sh ]; then chmod +x ./setup.sh && bash -c ./setup.sh; fi

if command -v conda >/dev/null 2>&1; then
  # set pip_interop_enabled to improve conda-pip interoperability. Conda can use
  # pip-installed packages to satisfy dependencies.
  # this option is only available after conda version 4.6.0
  # "|| true" ignores the error when the option is not found, for older conda version
  # This is commented out due to a bug with conda's implementation, we should revisit
  # after conda remove the experimental flag on pip_interop_enabled option
  # See more details on https://github.com/bentoml/BentoML/pull/1012
  # conda config --set pip_interop_enabled True || true

  echo "Updating conda base environment with environment.yml"
  conda env update -n base -f ./environment.yml
else
  echo "Warning: conda command not found, skipping dependencies in environment.yml"
fi

# Install PyPI packages specified in requirements.txt
pip install -r ./requirements.txt --no-cache-dir

# install sdist or wheel format archives stored under bundled_pip_dependencies directory
for filename in ./bundled_pip_dependencies/*.tar.gz; do
  [ -e "$filename" ] || continue
  pip install -U "$filename" --no-cache-dir
done
