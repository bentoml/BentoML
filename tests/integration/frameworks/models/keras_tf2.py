from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.keras


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


input_data = [[1, 2, 3, 4, 5]]
res = KerasSequentialModel().predict(np.array(input_data))
input_array = np.array(input_data, dtype="float64")
input_array_i32 = np.array(input_data, dtype="int64")
input_tensor = tf.constant(input_data, dtype=tf.float64)
input_tensor_f32 = tf.constant(input_data, dtype=tf.float32)

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


models: list[FrameworkTestModel] = [native_sequential_model]
