#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

# Install Spacy
pip install spacy==3.1.2 pyyaml
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tensorflow==2.6.0
python -m spacy download en_core_web_sm

pytest "$GIT_ROOT"/tests/integration/frameworks/spacy --cov=bentoml --cov-config=.coveragerc

test $error = 0 # Return non-zero if pytest failed