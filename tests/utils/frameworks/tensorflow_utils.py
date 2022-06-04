import re
import string

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


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


class NativeModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="inputs")]
    )
    def __call__(self, inputs):
        return self.dense(inputs)


class NativeRaggedModel(NativeModel):
    @tf.function(
        input_signature=[
            tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float64, 1, tf.int64)
        ]
    )
    def __call__(self, inputs):
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
