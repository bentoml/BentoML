# pylint: disable=redefined-outer-name
import json
import tempfile

import pytest
import tensorflow as tf

from tests import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server_docker_container,
)

test_data = [[1.1, 2.2]]
test_tensor = tf.constant(test_data)


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

        # training operartion
        train_op = tf.train.AdamOptimizer().minimize(loss)

        return {"p": p, "loss": loss, "train_op": train_op, "X": X, "y": y}

    tf.reset_default_graph()
    cnn_model = cnn_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            prediction = sess.run(cnn_model["p"], {cnn_model["X"]: test_data})
            print(prediction)

            inputs = {"X": cnn_model["X"]}
            outputs = {"prediction": cnn_model["p"]}

            tf.saved_model.simple_save(sess, temp_dir, inputs=inputs, outputs=outputs)
        yield temp_dir


@pytest.fixture(scope="session")
def svc(tf1_model_path):
    """Return a TensorFlow1 BentoService."""
    # When the ExampleBentoService got saved and loaded again in the test, the
    # two class attribute below got set to the loaded BentoService class.
    # Resetting it here so it does not effect other tests
    from tests import Tensorflow1Classifier

    Tensorflow1Classifier._bento_service_bundle_path = None
    Tensorflow1Classifier._bento_service_bundle_version = None

    svc = Tensorflow1Classifier()
    svc.pack("model", tf1_model_path)
    return svc


@pytest.fixture(scope="session")
def image(svc, clean_context):
    with export_service_bundle(svc) as saved_path:
        yield clean_context.enter_context(build_api_server_docker_image(saved_path))


@pytest.fixture(scope="module")
def host(image):
    with run_api_server_docker_container(image, timeout=500) as host:
        yield host


@pytest.mark.asyncio
async def test_tensorflow_1_artifact_with_docker(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps({"instances": test_data}),
        assert_status=200,
        assert_data=b"[0]",
    )
