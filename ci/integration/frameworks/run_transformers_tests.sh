#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

pip install transformers==4.9.2
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install jax==0.2.19 jaxlib==0.1.70 flax==0.3.4
pip install tensorflow==2.5.0 importlib_metadata
pytest -s "$GIT_ROOT"/tests/integration/frameworks/test_transformers_impl.py --cov=bentoml --cov-config=.coveragerc --runslow

test $error = 0 # Return non-zero if pytest failed