from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.tensorflow

backward_compatible = True


class NativeModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="inputs")
        ]
    )
    def __call__(self, inputs):
        return self.dense(inputs)

    @tf.function(
        input_signature=[
            tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float64, 1, tf.int64)
        ]
    )
    def predict_ragged(self, inputs):
        inputs = inputs.to_tensor(shape=[None, 5], default_value=0)
        return self.dense(inputs)


class MultiInputModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda tensor: tf.matmul(tensor, self.weights)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="x1"),
            tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="x2"),
            tf.TensorSpec(shape=(), dtype=tf.float64, name="factor"),
        ]
    )
    def __call__(self, x1: tf.Tensor, x2: tf.Tensor, factor: tf.Tensor):
        return self.dense(x1 + x2 * factor)


def make_keras_sequential_model() -> tf.keras.models.Model:
    net = keras.models.Sequential(
        (
            keras.layers.Dense(
                units=1,
                input_shape=(5,),
                dtype=tf.float64,
                use_bias=False,
                kernel_initializer=keras.initializers.Ones(),
            ),
        )
    )
    opt = keras.optimizers.Adam(0.002, 0.5)
    net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    # net.fit(x=input_tensor, y=tf.constant([[15.0]]))
    return net


def make_keras_functional_model() -> tf.keras.Model:
    x = tf.keras.layers.Input((5,), dtype=tf.float64, name="x")
    y = tf.keras.layers.Dense(
        6,
        name="out",
        kernel_initializer=tf.keras.initializers.Ones(),
    )(x)
    return tf.keras.Model(inputs=x, outputs=y)


input_data = [[1, 2, 3, 4, 5]]
input_array = np.array(input_data, dtype="float64")
input_array_i32 = np.array(input_data, dtype="int64")
input_tensor = tf.constant(input_data, dtype=tf.float64)
input_tensor_f32 = tf.constant(input_data, dtype=tf.float32)

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor: "tf_ext.TensorLike" = tf.ragged.constant(ragged_data, dtype=tf.float64)


native_multi_input_model = FrameworkTestModel(
    name="tf2",
    model=MultiInputModel(),
    configurations=[
        Config(
            load_kwargs={
                "partial_kwargs": {
                    "__call__": {"factor": tf.constant(3.0, dtype=tf.float64)}
                }
            },
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[i, i],
                        expected=lambda out: np.isclose(out, [[60.0]]).all(),
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

native_model = FrameworkTestModel(
    name="tf2",
    model=NativeModel(),
    configurations=[
        Config(
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[i],
                        expected=lambda out: np.isclose(out, [[15.0]]).all(),
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
        Config(
            test_inputs={
                "predict_ragged": [
                    Input(
                        input_args=[ragged_tensor],
                        expected=lambda out: np.isclose(out, [[15.0]] * 3).all(),
                    ),
                ],
            },
        ),
    ],
)

keras_models = [
    FrameworkTestModel(
        name="tf2",
        model=model,
        save_kwargs={"signatures": {"__call__": {"batchable": True, "batch_dim": 0}}},
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[input_tensor],
                            expected=lambda out: np.isclose(out, [[15.0]]).all(),
                        ),
                        Input(
                            input_args=[input_array],
                            expected=lambda out: np.isclose(out, [[15.0]]).all(),
                        ),
                        Input(
                            input_args=[input_data],
                            expected=lambda out: np.isclose(out, [[15.0]]).all(),
                        ),
                    ],
                },
            ),
        ],
    )
    for model in [
        make_keras_functional_model(),
        make_keras_sequential_model(),
    ]
]
models: list[FrameworkTestModel] = keras_models + [
    native_model,
    native_multi_input_model,
]
