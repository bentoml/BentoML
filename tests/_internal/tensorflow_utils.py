import tensorflow as tf
import tensorflow.keras as keras

test_data = [1, 2, 3, 4, 5]


def custom_activation(x):
    return tf.nn.tanh(x) ** 2


class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = tf.Variable(units, name="units")

    def call(self, inputs, training=False, **kwargs):
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
