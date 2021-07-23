import os

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras

from bentoml.keras import KerasModel
from tests._internal.frameworks.tensorflow_utils import (
    CustomLayer,
    KerasSequentialModel,
    custom_activation,
    test_data,
)

TF2 = tf.__version__.startswith("2")


def predict_assert_equal(m1: keras.Model, m2: keras.Model):
    t_data = np.array([test_data])
    assert m1.predict(t_data) == m2.predict(t_data)


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (KerasSequentialModel(), {"store_as_json": True, "custom_objects": None}),
        (KerasSequentialModel(), {"store_as_json": False, "custom_objects": None}),
        (
            KerasSequentialModel(),
            {
                "store_as_json": False,
                "custom_objects": {
                    "CustomLayer": CustomLayer,
                    "custom_activation": custom_activation,
                },
            },
        ),
    ],
)
def test_keras_save_load(model, kwargs, tmpdir):
    KerasModel(model, **kwargs).save(tmpdir)
    if kwargs["custom_objects"]:
        assert os.path.exists(KerasModel.get_path(tmpdir, "_custom_objects.pkl"))
    if kwargs["store_as_json"]:
        assert os.path.exists(KerasModel.get_path(tmpdir, "_json.json"))
        assert os.path.exists(KerasModel.get_path(tmpdir, "_weights.hdf5"))
    else:
        assert os.path.exists(KerasModel.get_path(tmpdir, ".h5"))
    if not TF2:
        # Initialize variables in the graph/model
        init = tf.compat.v1.global_variables_initializer()
        KerasModel.sess.run(init)
        with KerasModel.sess.as_default():
            keras_loaded = KerasModel.load(tmpdir)
            predict_assert_equal(keras_loaded, model)
    else:
        keras_loaded = KerasModel.load(tmpdir)
        predict_assert_equal(keras_loaded, model)
