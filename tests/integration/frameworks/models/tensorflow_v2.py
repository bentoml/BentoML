from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.tensorflow


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


"""
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


def KerasNLPModel() -> keras.models.Model:
    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

    def custom_standardization(input_data: str) -> tf.Tensor:
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape(string.punctuation), ""
        )

    max_features = 20000
    embedding_dims = 50

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=400,
    )

    # A text input with preprocessing layers
    text_input = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorize_layer(text_input)
    x = keras.layers.Embedding(max_features + 1, embedding_dims)(x)
    x = keras.layers.Dropout(0.2)(x)

    # Conv1D + global max pooling
    x = keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=1)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = keras.Model(text_input, predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def test_tensorflow_v2_multi_args():
    model_class = MultiInputModel()
    bento_model = bentoml.tensorflow.save_model(MODEL_NAME, model_class)

    partial_kwargs1 = {"__call__": {"factor": tf.constant(3.0, dtype=tf.float64)}}
    runner1 = bento_model.with_options(partial_kwargs=partial_kwargs1).to_runner()

    partial_kwargs2 = {"__call__": {"factor": tf.constant(2.0, dtype=tf.float64)}}
    runner2 = bento_model.with_options(partial_kwargs=partial_kwargs2).to_runner()

    runner1.init_local()
    runner2.init_local()
    assert runner1.run(native_data, native_data) == np.array([[60.0]])
    assert runner2.run(native_data, native_data) == np.array([[45.0]])
"""


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
        # save_kwargs={"signature": {"__call__": {"batchable": True, "batchdim": 0}}},
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
