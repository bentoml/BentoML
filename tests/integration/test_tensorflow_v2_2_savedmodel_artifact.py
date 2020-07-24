import pytest

import tensorflow as tf

import bentoml
from tests.bento_service_examples.tensorflow_classifier import TensorflowClassifier


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
    TensorflowClassifier._bento_service_bundle_path = None
    TensorflowClassifier._bento_service_bundle_version = None

    svc = TensorflowClassifier()
    model = Tensorflow2Model()
    svc.pack('model', model)

    return svc


@pytest.fixture()
def tf2_svc_saved_dir(tmpdir, tf2_svc):
    """Save a TensorFlow2 BentoService and return the saved directory."""
    tmpdir = str(tmpdir)
    tf2_svc.save_to_dir(tmpdir)

    return tmpdir


@pytest.fixture()
def tf2_svc_loaded(tf2_svc_saved_dir):
    """Return a TensorFlow2 BentoService that has been saved and loaded."""
    return bentoml.load(tf2_svc_saved_dir)


test_df = tf.expand_dims(tf.constant([1, 2, 3, 4, 5]), 0)


def test_tensorflow_2_artifact(tf2_svc):
    assert tf2_svc.predict(test_df) == 15.0,\
        'Inference on unsaved TF2 artifact does not match expected'


def test_tensorflow_2_artifact_loaded(tf2_svc_loaded):
    assert tf2_svc_loaded.predict(test_df) == 15.0,\
        'Inference on saved and loaded TF2 artifact does not match expected'
