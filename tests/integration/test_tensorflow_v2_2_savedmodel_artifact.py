# pylint: disable=redefined-outer-name
import json

import numpy as np
import pytest
import tensorflow as tf

import bentoml
from tests.bento_service_examples.tensorflow_classifier import Tensorflow2Classifier
from tests.integration.api_server.conftest import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server_docker_container,
)

test_data = [[1, 2, 3, 4, 5]]
test_tensor = tf.constant(np.asfarray(test_data))

import contextlib


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


class TfKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple linear layer which sums the inputs
        self.dense = tf.keras.layers.Dense(
            units=1,
            input_shape=(5,),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
        )

    def call(self, inputs):
        return self.dense(inputs)


class TfNativeModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        super(TfNativeModel, self).__init__()
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64, name='inputs')]
    )
    def __call__(self, inputs):
        return self.dense(inputs)


@pytest.fixture(params=[TfKerasModel, TfNativeModel], scope="session")
def model_class(request):
    return request.param


@pytest.fixture(scope="session")
def tf2_svc(model_class):
    """Return a TensorFlow2 BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    Tensorflow2Classifier._bento_service_bundle_path = None
    Tensorflow2Classifier._bento_service_bundle_version = None

    svc = Tensorflow2Classifier()
    model = model_class()
    model(test_tensor)
    svc.pack('model', model)
    return svc


@pytest.fixture(params=[False, True], scope="module")
def enable_microbatch(request):
    return request.param


@pytest.fixture(scope="module")
def tf2_host(tf2_svc, enable_microbatch, clean_context):
    with export_service_bundle(tf2_svc) as saved_path:
        server_image = clean_context.enter_context(
            build_api_server_docker_image(saved_path)
        )
        with run_api_server_docker_container(
            server_image, enable_microbatch=enable_microbatch, timeout=500
        ) as host:
            yield host


def test_tensorflow_2_artifact(tf2_svc):
    assert (
        tf2_svc.predict(test_tensor) == 15.0
    ), 'Inference on unsaved TF2 artifact does not match expected'


def test_tensorflow_2_artifact_loaded(tf2_svc):
    with export_service_bundle(tf2_svc) as saved_path:
        tf2_svc_loaded = bentoml.load(saved_path)
        assert (
            tf2_svc.predict(test_tensor) == tf2_svc_loaded.predict(test_tensor) == 15.0
        ), 'Inference on saved and loaded TF2 artifact does not match expected'


@pytest.mark.asyncio
async def test_tensorflow_2_artifact_with_docker(tf2_host):
    await pytest.assert_request(
        "POST",
        f"http://{tf2_host}/predict",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps({"instances": test_data}),
        assert_status=200,
        assert_data=b'[[15.0]]',
    )
