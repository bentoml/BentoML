import os

import numpy as np
import pytest
import tensorflow as tf

from bentoml.tensorflow import TensorflowModel
from tests._internal.frameworks.tensorflow_utils import (
    KerasSequentialModel,
    NativeModel,
    NativeRaggedModel,
)

native_data = [[1, 2, 3, 4, 5]]
native_tensor = tf.constant(np.asfarray(native_data))

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


def predict__model(model, tensor):
    return model(tensor)


@pytest.mark.parametrize(
    "model_class, input_type, predict_fn",
    [
        (KerasSequentialModel(), native_tensor, predict__model),
        (NativeModel(), native_tensor, predict__model),
        (NativeRaggedModel(), ragged_tensor, predict__model),
    ],
)
def test_tensorflow_v2_save_load(model_class, input_type, predict_fn, tmpdir):
    TensorflowModel(model_class).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
    tf2_loaded = TensorflowModel.load(tmpdir)
    predict_fn(tf2_loaded, input_type)
    comparison = predict_fn(tf2_loaded, input_type) == predict_fn(
        model_class, input_type
    )
    assert all(comparison)
