import tempfile

import pytest
import tensorflow as tf

import bentoml.tensorflow
from tests._internal.helpers import assert_have_file_extension

test_data = [[1.1, 2.2]]


def predict__model(model, tensor):
    pred_func = model.signatures["serving_default"]
    return pred_func(tensor)["prediction"]


@pytest.fixture(scope="session")
def tf1_model_path():
    # Function below builds model graph
    def cnn_model_fn():
        X = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="X")

        # dense layer
        inter1 = tf.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
        p = tf.argmax(input=inter1, axis=1)

        # loss
        y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
        loss = tf.losses.softmax_cross_entropy(y, inter1)

        # training operation
        train_op = tf.train.AdamOptimizer().minimize(loss)

        return {"p": p, "loss": loss, "train_op": train_op, "X": X, "y": y}

    tf.reset_default_graph()
    cnn_model = cnn_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(cnn_model["p"], {cnn_model["X"]: test_data})

            inputs = {"X": cnn_model["X"]}
            outputs = {"prediction": cnn_model["p"]}

            tf.saved_model.simple_save(sess, temp_dir, inputs=inputs, outputs=outputs)
        yield temp_dir


def test_tensorflow_v1_save_load(tf1_model_path, modelstore):
    tag = bentoml.tensorflow.save(
        "tensorflow_test", tf1_model_path, model_store=modelstore
    )
    model_info = modelstore.get(tag)
    assert_have_file_extension(model_info.path, ".pb")
    tf1_loaded = bentoml.tensorflow.load("tensorflow_test", model_store=modelstore)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prediction: tf.Tensor = predict__model(tf1_loaded, tf.constant(test_data))
        assert prediction.shape == (1,)
