#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

# Use conda python
export PATH="${HOME}/miniconda/bin:${PATH}"
hash -r

# Install PyTorch (for training a model) and coremltools (to convert trained PyTorch model to CoreML).
# On Linux we'd might install torch==1.5.0+cpu torchvision==0.6.0+cpu but this runs on Mac.
pip install coremltools==4.0b2 torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

conda install -c conda-forge --yes ruamel.yaml

pytest "$GIT_ROOT"/tests/integration/test_coreml_model_artifact.py --cov=bentoml --cov-config=.coveragerc

test $error = 0 # Return non-zero if pytest failed
