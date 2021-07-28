import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from bentoml.onnxmlir import ONNXMlirModel
from tests._internal.frameworks.tensorflow_utils import NativeModel
from tests._internal.helpers import assert_have_file_extension

try:
    # this has to be able to find the arch and OS specific PyRuntime .so file
    from PyRuntime import ExecutionSession
except ImportError:
    raise Exception("PyRuntime package library must be in python path")

sys.path.append("/workdir/onnx-mlir/build/Debug/lib/")

test_data = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
test_df = pd.DataFrame(test_data, columns=["A", "B", "C", "D", "E"])
test_tensor = np.asfarray(test_data)


def predict_df(inference_sess: ExecutionSession, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float64)
    return inference_sess.run(input_data)


@pytest.fixture()
def tensorflow_model(tmpdir):
    model = NativeModel()
    tf.saved_model.save(model, tmpdir)


@pytest.fixture()
def convert_to_onnx(tensorflow_model, tmpdir):
    model_path = os.path.join(tmpdir, "model.onnx")
    command = [
        "python",
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        ".",
        "--output",
        model_path,
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir, text=True
    )
    stdout, stderr = docker_proc.communicate()
    assert "ONNX model is saved" in stderr, "Failed to convert TF model"


@pytest.fixture()
def compile_model(convert_to_onnx, tmpdir):
    sys.path.append("/workdir/onnx-mlir/build/Debug/lib/")
    model_location = os.path.join(tmpdir, "model.onnx")
    command = ["./onnx-mlir", "--EmitLib", model_location]
    onnx_mlir_loc = "/workdir/onnx-mlir/build/Debug/bin"

    docker_proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=onnx_mlir_loc,
    )
    stdout, stderr = docker_proc.communicate()
    # returns something like: 'Shared library model.so has been compiled.'
    assert "has been compiled" in stdout, "Failed to compile model"


def test_onnxmlir_save_load(compile_model, tmpdir):
    model = os.path.join(tmpdir, "model.so")
    ONNXMlirModel(model).save(tmpdir)
    assert_have_file_extension(tmpdir, ".so")

    onnxmlir_loaded = ONNXMlirModel.load(tmpdir)
    # fmt: off
    assert predict_df(ExecutionSession(model, "run_main_graph"), test_df) == predict_df(onnxmlir_loaded, test_df)  # noqa
    # fmt: on
