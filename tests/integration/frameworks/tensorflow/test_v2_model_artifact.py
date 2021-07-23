import os

import numpy as np
import pytest
import tensorflow as tf

from bentoml.tensorflow import TensorflowModel
from tests._internal.frameworks.tensorflow_utils import (
    KerasModel,
    NativeModel,
    NativeRaggedModel,
)

native_data = [[1, 2, 3, 4, 5]]
native_tensor = tf.constant(np.asfarray(native_data))

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


def predict_native_model(model, tensor):
    return model(tensor)


def predict_ragged_model(model, tensor):
    tensor = tf.ragged.constant(tensor, dtype=tf.float64)
    return model(tensor)


@pytest.mark.parametrize(
    "model_class, input_type, predict_fn",
    [
        (KerasModel, native_tensor, predict_native_model),
        (NativeModel, native_tensor, predict_native_model),
        (NativeRaggedModel, ragged_tensor, predict_ragged_model),
    ],
)
def test_tensorflow_v2_save_load(model_class, input_type, predict_fn, tmpdir):
    TensorflowModel(model_class()).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
    tf2_loaded = TensorflowModel.load(tmpdir)
    assert predict_fn(tf2_loaded, input_type) == predict_fn(model_class(), input_type)
