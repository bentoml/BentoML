# pylint: disable=redefined-outer-name
import asyncio
import json

import numpy as np
import pytest
import tensorflow as tf

import bentoml
from tests.bento_services.tensorflow_classifier import Tensorflow2Classifier
from tests.integration.utils import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server_docker_container,
)

test_data = [[1, 2, 3, 4, 5]]
test_tensor = tf.constant(np.asfarray(test_data))

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


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
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64, name="inputs")]
    )
    def __call__(self, inputs):
        return self.dense(inputs)


class TfNativeModelWithRagged(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[
            tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float64, 1, tf.int64)
        ]
    )
    def __call__(self, inputs):
        inputs = inputs.to_tensor(shape=[None, 5], default_value=0)
        return self.dense(inputs)


@pytest.fixture(scope="session")
def svc():
    """Return a TensorFlow2 BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    Tensorflow2Classifier._bento_service_bundle_path = None
    Tensorflow2Classifier._bento_service_bundle_version = None

    svc = Tensorflow2Classifier()

    model1 = TfKerasModel()
    model1(test_tensor)
    svc.pack("model1", model1)

    model2 = TfNativeModel()
    model2(test_tensor)
    svc.pack("model2", model2)

    model3 = TfNativeModelWithRagged()
    model3(ragged_tensor)
    svc.pack("model3", model3)

    return svc


@pytest.fixture(scope="session")
def image(svc, clean_context):
    with export_service_bundle(svc) as saved_path:
        yield clean_context.enter_context(build_api_server_docker_image(saved_path))


@pytest.fixture(scope="module")
def host(image):
    with run_api_server_docker_container(image, timeout=500) as host:
        yield host


def test_tensorflow_2_artifact(svc):
    assert (
        svc.predict1(test_tensor) == 15.0
    ), "Inference on unsaved TF2 artifact does not match expected"

    assert (
        svc.predict2(test_tensor) == 15.0
    ), "Inference on unsaved TF2 artifact does not match expected"

    assert (
        (svc.predict3(ragged_data) == 15.0).numpy().all()
    ), "Inference on unsaved TF2 artifact does not match expected"


def test_tensorflow_2_artifact_loaded(svc):
    with export_service_bundle(svc) as saved_path:
        svc_loaded = bentoml.load(saved_path)
        assert (
            svc_loaded.predict1(test_tensor) == 15.0
        ), "Inference on saved and loaded TF2 artifact does not match expected"
        assert (
            svc_loaded.predict2(test_tensor) == 15.0
        ), "Inference on saved and loaded TF2 artifact does not match expected"
        assert (
            (svc_loaded.predict3(ragged_data) == 15.0).numpy().all()
        ), "Inference on saved and loaded TF2 artifact does not match expected"


@pytest.mark.asyncio
async def test_tensorflow_2_artifact_with_docker(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict1",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps({"instances": test_data}),
        assert_status=200,
        assert_data=b"[[15.0]]",
    )
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict2",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps({"instances": test_data}),
        assert_status=200,
        assert_data=b"[[15.0]]",
    )
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/predict3",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps(i),
            assert_status=200,
            assert_data=b"[15.0]",
        )
        for i in ragged_data
    )
    await asyncio.gather(*tasks)
