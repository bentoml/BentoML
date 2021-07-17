# pylint: disable=redefined-outer-name

import json

import keras
import numpy as np
import pytest
import tensorflow as tf

import bentoml
from tests import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server_docker_container,
)

TF2 = tf.__version__.startswith("2")

if TF2:
    from tests import KerasClassifier
else:
    from tests import KerasClassifier

test_data = [1, 2, 3, 4, 5]


@pytest.fixture(params=[tf.keras, keras], scope="session")
def keras_model(request):
    ke = request.param
    net = ke.Sequential(
        (
            ke.layers.Dense(
                units=1,
                input_shape=(5,),
                use_bias=False,
                kernel_initializer=ke.initializers.Ones(),
            ),
        )
    )
    net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return net


@pytest.fixture(scope="session")
def svc(keras_model):
    """Return a TensorFlow2 BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests

    KerasClassifier._bento_service_bundle_path = None
    KerasClassifier._bento_service_bundle_version = None

    svc = KerasClassifier()
    keras_model.predict(np.array([test_data]))
    svc.pack("model", keras_model)
    svc.pack("model2", keras_model)
    return svc


@pytest.fixture(scope="session")
def image(svc, clean_context):
    with export_service_bundle(svc) as saved_path:
        yield clean_context.enter_context(build_api_server_docker_image(saved_path))


@pytest.fixture(scope="module")
def host(image):
    with run_api_server_docker_container(image, timeout=500) as host:
        yield host


def test_keras_artifact(svc):
    assert svc.predict([test_data]) == [
        15.0
    ], "Inference on unsaved Keras artifact does not match expected"
    assert svc.predict2([test_data]) == [
        15.0
    ], "Inference on unsaved Keras artifact does not match expected"


def test_keras_artifact_loaded(svc):
    with export_service_bundle(svc) as saved_path:
        loaded = bentoml.load(saved_path)
        assert (
            loaded.predict([test_data]) == 15.0
        ), "Inference on saved and loaded Keras artifact does not match expected"
        assert (
            loaded.predict2([test_data]) == 15.0
        ), "Inference on saved and loaded Keras artifact does not match expected"


@pytest.mark.asyncio
async def test_keras_artifact_with_docker(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps(test_data),
        assert_status=200,
        assert_data=b"[15.0]",
    )
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict2",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps(test_data),
        assert_status=200,
        assert_data=b"[15.0]",
    )
