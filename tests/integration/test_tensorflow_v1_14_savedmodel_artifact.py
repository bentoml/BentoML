import pytest

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

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


class Tensorflow1Model(tf.keras.Model):
    def __init__(self):
        super(Tensorflow1Model, self).__init__()
        self.sess = tf.compat.v1.Session()
        self.X = tf.compat.v1.placeholder(tf.float32)

        # Simple linear layer which sums the inputs
        self.w = tf.expand_dims(tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0]), 0)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def __call__(self, inputs):
        return self.sess.run(tf.matmul(inputs, self.w, transpose_b=True))


test_df = tf.expand_dims(tf.constant([1, 2, 3, 4, 5], dtype=tf.float32), 0)



def test_tensorflow_1_artifact_pack(tensorflow_classifier_class):
    svc = tensorflow_classifier_class()
    model = Tensorflow1Model()
    svc.pack('model', model)
    assert svc.predict(test_df) == 15.0, 'Run inference before save the artifact'

    # Note: loaded_svc.predict() doesn't work, see this warning:
    # https://github.com/bentoml/BentoML/pull/421/commits/6e6c984b4967042f80f9f4278dfb4e4e5d57af8b#diff-53660cf1c2406d692f921b5194965ec0R123
    # This is a potential work-around, but I can't yet get it to work.
    #     wrapped_predict = loaded_svc.signatures[
    #         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # saved_path = svc.save()
    # loaded_svc = bentoml.load(saved_path)
    # assert loaded_svc.predict(test_df) == 15.0, 'Run inference from saved artifact'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.dangerously_delete_bento(svc.name, svc.version)
