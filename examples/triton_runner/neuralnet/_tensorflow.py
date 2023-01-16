from __future__ import annotations

import argparse

import tensorflow as tf
import keras.metrics
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten

import bentoml


class Net(tf.Module):
    def __init__(self):
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def __call__(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()

    # Import data
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train.reshape(60000, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(10000, 28, 28, 1).astype("float32")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # Create an instance of the model
    model = Net()

    optimizer = tf.keras.optimizers.Adam()
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = keras.metrics.Mean(name="train_loss")
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = keras.metrics.Mean(name="test_loss")
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {train_loss.result()}, "
            f"Accuracy: {train_accuracy.result() * 100}, "
            f"Test Loss: {test_loss.result()}, "
            f"Test Accuracy: {test_accuracy.result() * 100}"
        )

    _ = bentoml.tensorflow.save_model(
        "tensorflow-mnist",
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
    )

    print("Saved model:", _)


if __name__ == "__main__":
    try:
        m = bentoml.models.get("tensorflow-mnist")
        print("Model exists:", m)
    except bentoml.exceptions.NotFound:
        main()
