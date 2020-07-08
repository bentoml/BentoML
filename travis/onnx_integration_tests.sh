#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

# Install required packages for onnx model artifacts test
pip install onnx onnxruntime skl2onnx
pytest "$GIT_ROOT"/tests/integration/test_onnx_model_artifact.py --cov=bentoml --cov-config=.coveragerc


test $error = 0 # Return non-zero if pytest failed
