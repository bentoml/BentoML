from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import keras
import numpy as np
import tensorflow as tf
import keras.layers
import keras.optimizers

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    from bentoml._internal import external_typing as ext
    from bentoml._internal.external_typing import tensorflow as tf_ext

framework = bentoml.keras

backward_compatible = True


def custom_activation(x: tf_ext.TensorLike) -> tf_ext.TensorLike:
    return tf.nn.tanh(x) ** 2


class CustomLayer(keras.layers.Layer):
    def __init__(self, units: int = 32, **kwargs: t.Any):
        super().__init__(**kwargs)
        self.units = tf.Variable(units, name="units")

    def call(self, inputs: t.Any, *args: t.Any, training: bool = False):
        if training:
            return inputs * self.units
        else:
            return inputs

    def get_config(self):
        config: dict[str, t.Any] = super().get_config()
        config.update({"units": self.units.numpy()})
        return config


def KerasSequentialModel() -> keras.models.Model:
    net = keras.models.Sequential(
        (
            keras.layers.Dense(
                units=1,
                input_shape=(5,),
                use_bias=False,
                kernel_initializer="Ones",
            ),
        )
    )
    opt = keras.optimizers.adam_v2.Adam(0.002, 0.5)
    net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return net


input_data = [[1, 2, 3, 4, 5]]
res: ext.NpNDArray = KerasSequentialModel().predict(np.array(input_data))
input_array = np.array(input_data, dtype="float64")
input_array_i32 = np.array(input_data, dtype="int64")
input_tensor: tf_ext.TensorLike = tf.constant(input_data, dtype=tf.float64)
input_tensor_f32: tf_ext.TensorLike = tf.constant(input_data, dtype=tf.float32)


native_sequential_model = FrameworkTestModel(
    name="keras_tf2",
    model=KerasSequentialModel(),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[i],
                        expected=lambda out: np.isclose(out, res).all(),
                    )
                    for i in [
                        input_tensor,
                        input_tensor_f32,
                        input_array,
                        input_array_i32,
                        input_data,
                    ]
                ],
            },
        ),
    ],
)


def KerasFunctionalModel() -> keras.models.Model:
    class FunctionalModel(keras.Model):
        def __init__(self, **kwargs: t.Any):
            super().__init__(**kwargs)
            self.dense_1 = keras.layers.Dense(30, activation="relu")
            self.dense_2 = keras.layers.Dense(10)

        @tf.function(
            input_signature=[tf.TensorSpec(shape=(None, 32), dtype=tf.float32)]
        )
        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            inputs = inputs[:30]
            x = self.dense_1(inputs)
            return self.dense_2(x)

    model = FunctionalModel()
    model.predict(tf.ones((1, 32)))
    return model


input_data2 = np.ones((1, 32))
functional_model = KerasFunctionalModel()
res2: ext.NpNDArray = functional_model.predict(np.array(input_data2))
input_array2 = np.array(input_data2, dtype="float64")
input_tensor2: ext.NpNDArray = tf.constant(input_data2, dtype=tf.float32)

native_functional_model = FrameworkTestModel(
    name="keras_tf2",
    model=functional_model,
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[i],
                        expected=lambda out: np.isclose(out, res2).all(),
                    )
                    for i in [
                        input_tensor2,
                        input_array2,
                        input_data2,
                    ]
                ],
            },
        ),
    ],
)

models: list[FrameworkTestModel] = [native_sequential_model, native_functional_model]
