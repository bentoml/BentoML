import pytest

import tensorflow as tf

import bentoml
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.tensorflow_classifier import TensorflowClassifier


@pytest.fixture()
def tensorflow_classifier_class():
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    TensorflowClassifier._bento_service_bundle_path = None
    TensorflowClassifier._bento_service_bundle_version = None
    return TensorflowClassifier


class TensorflowModel(tf.keras.Model):
    def __init__(self):
        super(TensorflowModel, self).__init__()
        # Simple linear layer which sums the inputs
        self.dense = tf.keras.layers.Dense(
            units=1,
            input_shape=(5,),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
        )

    def call(self, inputs):
        return self.dense(inputs)


test_df = tf.expand_dims(tf.constant([1, 2, 3, 4, 5]), 0)


def test_tensorflow_artifact_pack(tensorflow_classifier_class):
    svc = tensorflow_classifier_class()
    model = TensorflowModel()
    svc.pack('model', model)
    assert svc.predict(test_df) == 15.0, 'Run inference before save the artifact'

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_df) == 15.0, 'Run inference from saved artifact'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)
