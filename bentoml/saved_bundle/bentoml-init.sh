#!/usr/bin/env bash
# Bash Script that installs the dependencies specified in the BentoService archive
#
# Usage:
#   * `bentoml-init.sh` to run the full script
#   * `bentoml-init.sh <step_name>` to run a specific step 
#      available steps: [setup.py python_version environment.yml requirement.txt bundled_pip_dependencies]

set -ex

# cd to the saved bundle directory
SAVED_BUNDLE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd $SAVED_BUNDLE_PATH

# Run the user defined setup.sh script if it is presented
if [ $# -eq 0 ] || [ $1 == "setup.py" ] ; then
  if [ -f ./setup.sh ]; then chmod +x ./setup.sh && bash -c ./setup.sh; fi
fi

# Check and install the right python version
if [ $# -eq 0 ] || [ $1 == "python_version" ] ; then
  if [ -f ./python_version ]; then
    PY_VERSION_SAVED=$(cat ./python_version)
    # remove PATCH version - since most patch version only contains backwards compatible
    # bug fixes and the BentoML defautl docker base image will include the latest
    # patch version of each Python minor release
    DESIRED_PY_VERSION=${PY_VERSION_SAVED:0:3} # returns 3.6, 3.7 or 3.8
    CURRENT_PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if [[ "$DESIRED_PY_VERSION" == "$CURRENT_PY_VERSION" ]]; then
      echo "Python Version in docker base image $CURRENT_PY_VERSION matches requirement python=$DESIRED_PY_VERSION. Skipping."
    else
      if command -v conda >/dev/null 2>&1; then
        echo "Installing python=$DESIRED_PY_VERSION with conda:"
        conda install -y -n base pkgs/main::python=$DESIRED_PY_VERSION pip
      else
        echo "WARNING: Python Version $DESIRED_PY_VERSION is required, but $CURRENT_PY_VERSION was found."
      fi
    fi
  fi
fi

if [ $# -eq 0 ] || [ $1 == "environment.yml" ] ; then
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
    conda clean --all
  else
    echo "WARNING: conda command not found, skipping conda dependencies in environment.yml"
  fi
fi

# Install PyPI packages specified in requirements.txt
if [ $# -eq 0 ] || [ $1 == "requirements.txt" ] ; then
  pip install -r ./requirements.txt --no-cache-dir $EXTRA_PIP_INSTALL_ARGS
fi

# Install additional python packages inside bundled pip dependencies directory
if [ $# -eq 0 ] || [ $1 == "bundled_pip_dependencies" ] ; then
  for filename in ./bundled_pip_dependencies/*; do
    [ -e "$filename" ] || continue
    pip install -U "$filename"
  done
fi
