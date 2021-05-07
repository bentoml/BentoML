import pytest
import numpy as np
import pandas
import tensorflow as tf
import subprocess
from tests.bento_service_examples.onnxmlir_classifier import OnnxMlirClassifier
from bentoml.yatai.client import YataiClient

test_data = [[1, 2, 3, 4, 5]]
test_df = pandas.DataFrame([[1, 2, 3, 4, 5]])
test_tensor = np.asfarray(test_data)


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


@pytest.fixture()
def onnxmlir_classifier_class():
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    OnnxMlirClassifier._bento_service_bundle_path = None
    OnnxMlirClassifier._bento_service_bundle_version = None
    return OnnxMlirClassifier()


@pytest.fixture()
def tensorflow_model(tmp_path_factory):
    model1 = TfNativeModel()
    tmpdir = str(tmp_path_factory.mktemp("tf2_model"))
    tf.saved_model.save(model1, tmpdir)
    return tmpdir


@pytest.fixture()
def convert_to_onnx(tensorflow_model, tmp_path_factory):
    tf_model = tensorflow_model
    tmpdir = str(tmp_path_factory.mktemp("onnx_model"))
    modelpath = tmpdir + '/model.onnx'
    command = [
        'python',
        '-m',
        'tf2onnx.convert',
        '--saved-model',
        '.',
        '--output',
        modelpath,
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tf_model, text=True
    )
    stdout, stderr = docker_proc.communicate()
    assert 'ONNX model is saved' in stderr, 'Failed to convert TF model'
    return tmpdir


@pytest.fixture()
def compile_model(convert_to_onnx, tmp_path_factory):
    onnxmodelloc = convert_to_onnx + '/model.onnx'
    command = ['./onnx-mlir', '--EmitLib', onnxmodelloc]
    onnx_mlir_loc = '/workdir/onnx-mlir/build/Debug/bin'

    ppwd = ['pwd']
    pwd_cmd = subprocess.Popen(
        ppwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=onnx_mlir_loc, text=True
    )
    stdouta, stderra = pwd_cmd.communicate()
    print(stdouta)
    print(stderra)

    docker_proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=onnx_mlir_loc,
    )
    stdout, stderr = docker_proc.communicate()
    # returns something like: 'Shared library model.so has been compiled.'
    assert 'has been compiled' in stdout, 'Failed to compile model'
    # modelname = 'model.so'
    return convert_to_onnx


@pytest.fixture()
def get_onnx_mlir_svc(compile_model, onnxmlir_classifier_class):
    svc = onnxmlir_classifier_class
    # need to check compile output location from compile_model
    model = compile_model + '/model.so'
    svc.pack('model', model)
    return svc


def test_onnxmlir_artifact(get_onnx_mlir_svc):
    svc = get_onnx_mlir_svc
    assert (
        svc.predict(test_df) == 15.0
    ), 'Inference on onnx-mlir artifact does not match expected'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
