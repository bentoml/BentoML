import json
import pytest

import tensorflow as tf

import bentoml
from tests.bento_service_examples.tensorflow_classifier import Tensorflow2Classifier
from tests.integration.api_server.conftest import (
    run_api_server_docker_container,
    build_api_server_docker_image,
)

test_data = [[1, 2, 3, 4, 5]]
test_tensor = tf.constant(test_data)


class Tensorflow2Model(tf.keras.Model):
    def __init__(self):
        super(Tensorflow2Model, self).__init__()
        # Simple linear layer which sums the inputs
        self.dense = tf.keras.layers.Dense(
            units=1,
            input_shape=(5,),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
        )

    def call(self, inputs):
        return self.dense(inputs)


@pytest.fixture(scope="module")
def tf2_svc():
    """Return a TensorFlow2 BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    Tensorflow2Classifier._bento_service_bundle_path = None
    Tensorflow2Classifier._bento_service_bundle_version = None

    svc = Tensorflow2Classifier()
    model = Tensorflow2Model()
    svc.pack('model', model)

    return svc


@pytest.fixture(scope="module")
def tf2_svc_saved_dir(tmp_path_factory, tf2_svc):
    """Save a TensorFlow2 BentoService and return the saved directory."""
    # Must be called at least once before saving so that layers are built
    # See: https://github.com/tensorflow/tensorflow/issues/37439
    tf2_svc.predict(test_tensor)

    tmpdir = str(tmp_path_factory.mktemp("tf2_svc"))
    tf2_svc.save_to_dir(tmpdir)

    return tmpdir


@pytest.fixture()
def tf2_svc_loaded(tf2_svc_saved_dir):
    """Return a TensorFlow2 BentoService that has been saved and loaded."""
    return bentoml.load(tf2_svc_saved_dir)


@pytest.fixture()
def tf2_image(tf2_svc_saved_dir):
    with build_api_server_docker_image(
        tf2_svc_saved_dir, "tf2_example_service"
    ) as image:
        yield image


@pytest.fixture()
def tf2_host(tf2_image):
    with run_api_server_docker_container(tf2_image, timeout=500) as host:
        yield host


def test_tensorflow_2_artifact(tf2_svc):
    assert (
        tf2_svc.predict(test_tensor) == 15.0
    ), 'Inference on unsaved TF2 artifact does not match expected'


def test_tensorflow_2_artifact_loaded(tf2_svc_loaded):
    assert (
        tf2_svc_loaded.predict(test_tensor) == 15.0
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
