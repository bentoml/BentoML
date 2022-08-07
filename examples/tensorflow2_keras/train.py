# pylint: disable=no-name-in-module,redefined-outer-name,abstract-method
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten

import bentoml

print("TensorFlow version:", tf.__version__)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)

    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # Create an instance of the model
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )

    bentoml.tensorflow.save_model(
        "tensorflow_mnist",
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    )
