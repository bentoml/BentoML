#!/usr/bin/env bash
set -x

# https://stackoverflow.com/questions/42218009/how-to-tell-if-any-command-in-bash-script-failed-non-zero-exit-status/42219754#42219754
# Set err to 1 if pytest failed.
error=0
trap 'error=1' ERR

cat /etc/os-release  

# Export path to onnx-mlir executable
export PATH="/workdir/onnx-mlir/build/Debug/bin/:${PATH}"
# Export path to PyRuntime so file
export PATH="/workdir/onnx-mlir/build/Debug/lib/:${PATH}"

PYTHONPATH="${PYTHONPATH}:/workdir/onnx-mlir/build/Debug/lib/"
export PYTHONPATH


GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit

python3 -m pip install pip --upgrade
python3 -m pip install tensorflow==2.2.0
python3 -m pip install -U tf2onnx

apt-get install curl

# Install required packages for onnx-mlir model artifacts test
# not here: pip install onnx onnxruntime skl2onnx
pytest "$GIT_ROOT"/tests/integration/test_onnxmlir_model_artifact.py --cov=bentoml --cov-config=.coveragerc


test $error = 0 # Return non-zero if pytest failed