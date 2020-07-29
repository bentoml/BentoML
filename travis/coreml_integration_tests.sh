#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

# Install miniconda - taken from https://stackoverflow.com/q/45257534/2064085
wget https://repo.continuum.io/miniconda/Miniconda3-latest-$MINICONDAVERSION-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r  # https://stackoverflow.com/q/45257534/2064085
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda config --add channels conda-forge
conda install -y python=3.6  # change base env Python to conda
# Useful for debugging any issues with conda
conda info -a
command -v python
command -v pip
python -V
pip -V

# Install PyTorch (for training a model) and coremltools (to convert trained PyTorch model to CoreML).
pip install coremltools==4.0b2 torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pytest "$GIT_ROOT"/tests/integration/test_coreml_model_artifact.py --cov=bentoml --cov-config=.coveragerc

test $error = 0 # Return non-zero if pytest failed
