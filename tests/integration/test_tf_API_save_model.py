import json

import numpy as np
import pytest
import tensorflow as tf

import bentoml
from bentoml.artifact import TensorflowSavedModelArtifact
from tests.bento_service_examples.tensorflow_classifier import Tensorflow2Classifier
from tests.integration.api_server.conftest import (
    build_api_server_docker_image,
    run_api_server_docker_container,
)

test_data = [[1, 2, 3, 4, 5]]
test_tensor = tf.constant(np.asfarray(test_data))
print(test_tensor.shape)


class CreateTfModel(tf.Module):
    def __init__(self):
        self.weights = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])
        super(CreateTfModel, self).__init__()
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function
    def __call__(self, inputs):
        return self.dense(inputs)


@pytest.fixture(scope="module")
def tf2_svc():
    """Return a TensorFlow2 BentoService."""
    Tensorflow2Classifier._bento_service_bundle_path = None
    Tensorflow2Classifier._bento_service_bundle_version = None

    svc = Tensorflow2Classifier()
    model = CreateTfModel()
    model(test_tensor)
    svc.pack('model', model)
    return svc


@pytest.fixture(scope="module")
def tf2_svc_saved_dir(tmp_path_factory, tf2_svc):
    """Save a Tensorflow2 BentoService and return the saved directory."""
    tf2_svc.predict(test_tensor)
    tmpdir = str(tmp_path_factory.mktemp("tf2_svc"))
    tf2_svc.save_to_dir(tmpdir)
    return tmpdir


@pytest.fixture()
def tf2_svc_loaded(tf2_svc_saved_dir):
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
    # assert (tf2_svc(test_tensor).shape == (1,2)), 'checkout the shape'
    # assert tuple(map(tuple, tf2_svc(test_tensor))) == tuple(map(tuple, [[95.0, 110.0]]))
    # assert test_tensorflow_2_artifact == tf.Tensor([[15.0]])

    assert tf2_svc.predict(test_tensor) == 15.0


def test_tensorflow_2_artifact_loaded(tf2_svc_loaded):
    # assert (tf2_svc_loaded(test_tensor).shape == (1,2)), 'checkout the shape'
    # assert tuple(map(tuple, tf2_svc_loaded(test_tensor))) == tuple(map(tuple, [[95.0, 110.0]]))
    assert tf2_svc_loaded(test_tensor) == 15.0
