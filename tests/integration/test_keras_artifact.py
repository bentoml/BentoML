# pylint: disable=redefined-outer-name

import contextlib
import json

import keras
import numpy as np
import pytest
import tensorflow as tf

import bentoml
from tests.bento_service_examples.keras_classifier import KerasClassifier
from tests.integration.api_server.conftest import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server_docker_container,
)

test_data = [1, 2, 3, 4, 5]


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.fixture(params=[tf.keras, keras], scope="session")
def model(request):
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
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return net


@pytest.fixture(scope="session")
def svc(model):
    """Return a TensorFlow2 BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    KerasClassifier._bento_service_bundle_path = None
    KerasClassifier._bento_service_bundle_version = None

    svc = KerasClassifier()
    model.predict(np.array([test_data]))
    svc.pack('model', model)
    return svc


@pytest.fixture(scope="session")
def image(svc, clean_context):
    with export_service_bundle(svc) as saved_path:
        yield clean_context.enter_context(build_api_server_docker_image(saved_path))


@pytest.fixture(params=[False, True], scope="module")
def enable_microbatch(request):
    return request.param


@pytest.fixture(scope="module")
def host(image, enable_microbatch):
    with run_api_server_docker_container(
        image, enable_microbatch=enable_microbatch, timeout=500
    ) as host:
        yield host


def test_keras_artifact(svc):
    assert svc.predict([test_data]) == [
        15.0
    ], 'Inference on unsaved Keras artifact does not match expected'


def test_keras_artifact_loaded(svc):
    with export_service_bundle(svc) as saved_path:
        loaded = bentoml.load(saved_path)
        assert (
            loaded.predict([test_data]) == 15.0
        ), 'Inference on saved and loaded Keras artifact does not match expected'


@pytest.mark.asyncio
async def test_keras_artifact_with_docker(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps(test_data),
        assert_status=200,
        assert_data=b'[15.0]',
    )
