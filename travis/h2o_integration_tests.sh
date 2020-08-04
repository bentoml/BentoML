#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit


# Install required packages for h2o model artifacts test
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda create --yes -n test python=$TRAVIS_PYTHON_VERSION
conda config --add channels h2oai
source activate test
conda install --yes openjdk

pip install h2o

pytest "$GIT_ROOT"/tests/integration/test_h2o_model_artifact.py --cov=bentoml --cov-config=.coveragerc

test $error = 0 # Return non-zero if pytest failed
