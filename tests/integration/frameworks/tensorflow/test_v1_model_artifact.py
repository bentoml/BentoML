import json
import tempfile
import typing as t

import pytest
import tensorflow

from bentoml.tensorflow import TensorflowModel

if tensorflow.__version__.startswith('2'):
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()
else:
    tf = tensorflow

tf.disable_eager_execution()


test_data: t.List[t.List[float]] = [[1.1, 2.2]]
test_tensor = tf.constant(test_data)


def predict_v1_tensor_model(model, tensor):
    pred_func = model.signatures['serving_default']
    return pred_func(tensor)['prediction']


# Function below builds model graph
def cnn_model_fn():
    X = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="X")

    # dense layer
    inter1 = tf.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
    p = tf.argmax(input=inter1, axis=1)

    # loss
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    loss = tf.losses.softmax_cross_entropy(y, inter1)

    # training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return {"p": p, "loss": loss, "train_op": train_op, "X": X, "y": y}


@pytest.fixture(scope="session")
def tf1_model_path():
    tf.reset_default_graph()
    cnn_model = cnn_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            prediction = sess.run(cnn_model["p"], {cnn_model["X"]: test_data})
            print(prediction)

            inputs = {"X": cnn_model['X']}
            outputs = {"prediction": cnn_model['p']}

            tf.saved_model.simple_save(sess, temp_dir, inputs=inputs, outputs=outputs)
        yield temp_dir


def test_tensorflow_v1_save_load(tf1_model_path, tmpdir):
    assert not TensorflowModel(tf1_model_path).save(tmpdir)
    tf1_loaded = TensorflowModel.load(tf1_model_path)
    print(predict_v1_tensor_model(tf1_loaded, test_tensor))
    assert predict_v1_tensor_model(tf1_loaded, test_tensor) == b'[0]'
