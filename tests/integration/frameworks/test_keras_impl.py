from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras

import bentoml
from tests.utils.helpers import assert_have_file_extension

MODEL_NAME = __name__.split(".")[-1]


def custom_activation(x):
    return tf.nn.tanh(x) ** 2


class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = tf.Variable(units, name="units")

    def call(self, inputs, training=False):
        if training:
            return inputs * self.units
        else:
            return inputs

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units.numpy()})
        return config


def KerasSequentialModel() -> keras.models.Model:
    net = keras.models.Sequential(
        (
            keras.layers.Dense(
                units=1,
                input_shape=(5,),
                use_bias=False,
                kernel_initializer=keras.initializers.Ones(),
            ),
        )
    )
    opt = keras.optimizers.Adam(0.002, 0.5)
    net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return net


test_data = [1, 2, 3, 4, 5]
res = KerasSequentialModel().predict(np.array([test_data]))


def predict_assert_equal(model: "keras.Model") -> None:
    t_data = np.array([test_data])
    assert model.predict(t_data) == res


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (
            KerasSequentialModel(),
            {},
        ),
        (
            KerasSequentialModel(),
            {
                "custom_objects": {
                    "CustomLayer": CustomLayer,
                    "custom_activation": custom_activation,
                },
            },
        ),
    ],
)
def test_keras_save_load(
    model: "keras.Model",
    kwargs: t.Dict[str, t.Any],
) -> None:
    bento_model = bentoml.keras.save_model(MODEL_NAME, model, **kwargs)
    assert bento_model == bentoml.keras.get(bento_model.tag)
    assert_have_file_extension(bento_model.path, ".pb")
    loaded = bentoml.keras.load_model(bento_model.tag)
    predict_assert_equal(loaded)


@pytest.mark.parametrize("signatures", [None, {"__call__": {"batchable": True}}])
def test_keras_run(signatures) -> None:
    model = KerasSequentialModel()
    bento_model = bentoml.keras.save_model(MODEL_NAME, model, signatures=signatures)
    runner = bento_model.to_runner()
    runner.init_local()

    # keras model's `__call__` will return a Tensor instead of
    # ndarray, here we test if we convert the Tensor to ndarray
    run_res = runner.run([test_data])
    assert isinstance(run_res, np.ndarray)
    assert np.array_equal(run_res, res)
