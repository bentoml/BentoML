import os

import numpy as np
import pytest
import tensorflow as tf

from bentoml.tensorflow import TensorflowModel

native_data = [[1, 2, 3, 4, 5]]
native_tensor = tf.constant(np.asfarray(native_data))

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


def predict_native_model(model, tensor):
    return model(tensor)


def predict_ragged_model(model, tensor):
    tensor = tf.ragged.constant(tensor, dtype=tf.float64)
    return model(tensor)


class KerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple linear layer which sums the inputs
        self.dense = tf.keras.layers.Dense(
            units=1,
            input_shape=(5,),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
        )

    def call(self, inputs, **kwargs):
        return self.dense(inputs)


class NativeModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64, name='inputs')]
    )
    def __call__(self, inputs):
        return self.dense(inputs)


class NativeRaggedModel(tf.Module):
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


@pytest.mark.parametrize(
    'model_class, input_type, predict_fn',
    [
        (KerasModel, native_tensor, predict_native_model),
        (NativeModel, native_tensor, predict_native_model),
        (NativeRaggedModel, ragged_tensor, predict_ragged_model),
    ],
)
def test_tensorflow_v2_save_load(model_class, input_type, predict_fn, tmpdir):
    TensorflowModel(model_class()).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, 'saved_model.pb'))
    tf2_loaded = TensorflowModel.load(tmpdir)
    assert predict_fn(tf2_loaded, input_type) == predict_fn(model_class(), input_type)
