#!/usr/bin/env bash
# Bash Script that installs the dependencies specified in the Bento
#
# Usage:
#   * `init.sh` to run the full script
#   * `init.sh <step_name>` to run a specific step
#      available steps: [ensure_python restore_conda_env install_pip_packages install_wheels user_setup_script]

set -ex

# Assert under the root of a Bento directory
if [ ! -d ./env  ]; then
    echo "init.sh must be executed from a Bento directory"
    exit 1
fi


# Check and install the right python version
if [ $# -eq 0 ] || [ $1 == "ensure_python" ] ; then
  if [ -f ./env/python/version.txt ]; then
    PY_VERSION_SAVED=$(cat ./env/python/version.txt)
    # remove PATCH version - since most patch version only contains backwards compatible
    # bug fixes and the BentoML defautl docker base image will include the latest
    # patch version of each Python minor release
    DESIRED_PY_VERSION=${PY_VERSION_SAVED:0:3} # returns 3.7, 3.8 or 3.9
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

if [ $# -eq 0 ] || [ $1 == "restore_conda_env" ] ; then
  if [ -d ./env/conda ] && [ -f ./env/conda/environment.yml ]; then
    if command -v conda >/dev/null 2>&1; then
      echo "Updating conda base environment with environment.yml"
      # set pip_interop_enabled to improve conda-pip interoperability. Conda can use
      # pip-installed packages to satisfy dependencies.
      # this option is only available after conda version 4.6.0
      # "|| true" ignores the error when the option is not found, for older conda version
      conda config --set pip_interop_enabled True || true
      conda env update -n base -f ./environment.yml
      conda clean --all
    else
      echo "WARNING: conda command not found, skipping conda dependencies in environment.yml"
    fi
  fi
  # Do nothing if not ./env/conda/environment.yml file is found
fi

# Install PyPI packages specified in requirements.lock.txt
if [ $# -eq 0 ] || [ $1 == "install_pip_packages" ] ; then
  if [ -f ./env/python/pip_args.txt ]; then
    EXTRA_PIP_INSTALL_ARGS=$(cat ./env/python/pip_args.txt)
  fi
  # BentoML by default generates two requirment files:
  #  - ./env/python/requirements.lock.txt: all dependencies locked to its version presented during `build`
  #  - ./env/python/requirements.txt: all dependecies as user specified in code or requirements.txt file
  # This build script will prioritize using `.lock.txt` file if it's available
  if [ -f ./env/python/requirements.lock.txt ]; then
    echo "Installing pip packages from 'requirements.lock.txt'.."
    pip install -r ./env/python/requirements.lock.txt --no-cache-dir $EXTRA_PIP_INSTALL_ARGS
  else
    if [ -f ./env/python/requirements.txt ]; then
      echo "Installing pip packages from 'requirements.txt'.."
      pip install -r ./env/python/requirements.txt --no-cache-dir $EXTRA_PIP_INSTALL_ARGS
    fi
  fi
fi

# Install wheels included in Bento
if [ $# -eq 0 ] || [ $1 == "install_wheels" ] ; then
  if [ -d ./env/python/wheels ]; then
    echo "Installing wheels.."
    pip install --no-cache-dir ./env/python/wheels/*.whl
  fi
fi

# Run the user defined setup_script if it is presented
if [ $# -eq 0 ] || [ $1 == "user_setup_script" ] ; then
  if [ -f ./env/docker/setup_script ]; then
    chmod +x ./env/docker/setup_script
    ./env/docker/setup_script;
  fi
fi