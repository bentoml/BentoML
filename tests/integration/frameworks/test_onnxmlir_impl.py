import os
import sys
import subprocess

import numpy as np
import pandas as pd
import psutil
import pytest

import bentoml
import tensorflow as tf
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.tensorflow_utils import NativeModel

try:
    # this has to be able to find the arch and OS specific PyRuntime .so file
    from PyRuntime import ExecutionSession
except ImportError:
    raise Exception("PyRuntime package library must be in PYTHONPATH")

sys.path.append("/workdir/onnx-mlir/build/Debug/lib/")

test_data = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
test_df = pd.DataFrame(test_data, columns=["A", "B", "C", "D", "E"])
test_tensor = np.asfarray(test_data)


def predict_df(inference_sess: ExecutionSession, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float64)
    return inference_sess.run(input_data)


@pytest.fixture(name="convert_to_onnx")
def fixture_convert_to_onnx(tmpdir):
    model = NativeModel()
    tf.saved_model.save(model, str(tmpdir))
    model_path = os.path.join(str(tmpdir), "model.onnx")
    command = [
        "python",
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        ".",
        "--output",
        model_path,
    ]
    docker_proc = subprocess.Popen(  # noqa
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir, text=True
    )
    _, stderr = docker_proc.communicate()
    assert "ONNX model is saved" in stderr, "Failed to convert TF model"


@pytest.fixture(name="compile_model")
def fixture_compile_model(convert_to_onnx, tmpdir):
    sys.path.append("/workdir/onnx-mlir/build/Debug/lib/")
    model_location = os.path.join(str(tmpdir), "model.onnx")
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


def test_onnxmlir_save_load(
    compile_model,
    tmpdir,
):  # noqa

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    model = os.path.join(tmpdir, "model.so")
    tag = bentoml.onnxmlir.save(
        "onnx_model_tests", model, labels=labels, custom_objects={"func": custom_f}
    )
    bentomodel = bentoml.models.get(tag)
    assert "compiled_path" in bentomodel.info.options
    assert_have_file_extension(str(bentomodel.path), ".so")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    session = bentoml.onnxmlir.load(tag)
    assert predict_df(session, test_df)[0] == np.array([[15.0]])


def test_onnxmlir_load_runner(
    compile_model,
    tmpdir,
):  # noqa
    model = os.path.join(tmpdir, "model.so")
    tag = bentoml.onnxmlir.save("onnx_model_tests", model)
    runner = bentoml.onnxmlir.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch(test_df.to_numpy().astype(np.float64))
    assert res[0] == np.array([[15.0]])
