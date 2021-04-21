import pytest
import numpy as np
import tensorflow as tf
import subprocess
from tests.bento_service_examples.onnxmlir_classifier import OnnxMlirClassifier
from bentoml.yatai.client import YataiClient

test_data = [[1, 2, 3, 4, 5]]
test_tensor = tf.constant(np.asfarray(test_data))

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


class TfNativeModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name='inputs')]
    )
    def __call__(self, inputs):
        return self.dense(inputs)


@pytest.fixture(scope="module")
def tensorflow_model(tmp_path_factory):
    model1 = TfNativeModel()
    tmpdir = str(tmp_path_factory.mktemp("tf2_model"))
    tf.saved_model.save(model1, tmpdir)
    return tmpdir


@pytest.fixture(scope="module")
def convert_to_onnx(tensorflow_model, tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("onnx_model"))
    modelpath = tmpdir + 'model.onnx'
    command = [
        'python -m tf2onnx.convert',
        '--saved-model',
        tensorflow_model,
        '--output',
        modelpath,
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout = docker_proc.stdout.read().decode('utf-8')
    assert 'model.onnx' in stdout, 'Failed to convert TF model'
    return modelpath


@pytest.fixture(scope="module")
def compile_model(convert_to_onnx, tmp_path_factory):
    command = [
        './onnx-mlir',
        '--EmitLib',
        convert_to_onnx,
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # should return something like: 'Shared library model.so has been compiled.'
    stdout = docker_proc.stdout.read().decode('utf-8')
    assert 'has been compiled' in stdout, 'Failed to compile model'
    modelname = 'model.so'
    return modelname


@pytest.fixture(scope="module")
def get_onnx_mlir_svc(compile_model):
    svc = OnnxMlirClassifier()
    # need to check compile output location from compile_model
    svc.pack('model', compile_model)
    return svc


def test_onnxmlir_artifact(get_onnx_mlir_svc):
    svc = get_onnx_mlir_svc()
    assert (
        get_onnx_mlir_svc.predict(test_tensor) == 15.0
    ), 'Inference on onnx-mlir artifact does not match expected'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
