import tempfile
from typing import Callable
from typing import Generator
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_hub as hub

import bentoml
from tests.utils.helpers import assert_have_file_extension

if TYPE_CHECKING:
    from bentoml._internal.models import ModelStore
    from bentoml._internal.external_typing import tensorflow as tf_ext

MODEL_NAME = __name__.split(".")[-1]

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

test_data = [[1.1, 2.2]]
test_tensor: "tf_ext.TensorLike" = tf.constant(test_data)

native_data = [[1, 2, 3, 4, 5]]
native_tensor: "tf_ext.TensorLike" = tf.constant(native_data, dtype=tf.float64)

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor: "tf_ext.TensorLike" = tf.ragged.constant(ragged_data, dtype=tf.float64)

sess = tf.compat.v1.Session()


def _model_dunder_call(
    model: "tf_ext.Module", tensor: "tf_ext.TensorLike"
) -> "tf_ext.TensorLike":
    pred_func = model.signatures["serving_default"]
    return pred_func(tensor)["prediction"]


@pytest.fixture(scope="session")
def tf1_model_path() -> Generator[str, None, None]:
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

    cnn_model = cnn_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            sess.run(cnn_model["p"], {cnn_model["X"]: test_data})

            inputs = {"X": cnn_model["X"]}
            outputs = {"prediction": cnn_model["p"]}

            tf.compat.v1.saved_model.simple_save(
                sess, temp_dir, inputs=inputs, outputs=outputs
            )
        yield temp_dir


@pytest.fixture(scope="session")
def tf1_multi_args_model_path() -> Generator[str, None, None]:
    def simple_model_fn():
        x1 = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="x1")
        x2 = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="x2")
        factor = tf.placeholder(shape=(), dtype=tf.float32, name="factor")

        w = tf.constant([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype=tf.float32)

        x = x1 + x2 * factor
        p = tf.matmul(x, w)
        return {"p": p, "x1": x1, "x2": x2, "factor": factor}

    simple_model = simple_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with sess.as_default():
            tf.enable_resource_variables()
            sess.run(tf.global_variables_initializer())
            inputs = {
                "x1": simple_model["x1"],
                "x2": simple_model["x2"],
                "factor": simple_model["factor"],
            }
            outputs = {"prediction": simple_model["p"]}

            tf.compat.v1.saved_model.simple_save(
                sess, temp_dir, inputs=inputs, outputs=outputs
            )

        yield temp_dir


def test_tensorflow_v1_save_load(
    tf1_model_path: Callable[[], Generator[str, None, None]], modelstore: "ModelStore"
):
    tag = bentoml.tensorflow_v1.save(
        "tensorflow_test", tf1_model_path, model_store=modelstore
    )
    model_info = modelstore.get(tag)
    assert_have_file_extension(model_info.path, ".pb")
    tf1_loaded = bentoml.tensorflow_v1.load("tensorflow_test", model_store=modelstore)
    with tf.get_default_graph().as_default():
        tf.global_variables_initializer()
        prediction = _model_dunder_call(tf1_loaded, test_tensor)
        assert prediction.shape == (1,)


def test_tensorflow_v1_setup_run_batch(
    tf1_model_path: Callable[[], Generator[str, None, None]], modelstore: "ModelStore"
):
    tag = bentoml.tensorflow_v1.save(
        "tensorflow_test", tf1_model_path, model_store=modelstore
    )
    runner = bentoml.tensorflow_v1.load_runner(tag, model_store=modelstore)

    with tf.get_default_graph().as_default():
        res = runner.run_batch(test_tensor)
        assert res.shape == (1,)


def test_tensorflow_v1_multi_args(
    tf1_multi_args_model_path: Callable[[], Generator[str, None, None]],
    modelstore: "ModelStore",
):
    tag = bentoml.tensorflow_v1.save(
        "tensorflow_test", tf1_multi_args_model_path, model_store=modelstore
    )
    x = tf.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=tf.float32)
    f1 = tf.convert_to_tensor(3.0, dtype=tf.float32)
    f2 = tf.convert_to_tensor(2.0, dtype=tf.float32)

    r1 = bentoml.tensorflow_v1.load_runner(
        tag,
        model_store=modelstore,
        partial_kwargs=dict(factor=f1),
    )

    r2 = bentoml.tensorflow_v1.load_runner(
        tag,
        model_store=modelstore,
        partial_kwargs=dict(factor=f2),
    )

    tf.global_variables_initializer()
    with sess.as_default():
        res = r1.run_batch(x1=x, x2=x)
        assert np.isclose(res.eval(), 60.0)

        res = r2.run_batch(x1=x, x2=x)
        assert np.isclose(res.eval(), 45.0)


def _plus_one_model_tf1() -> "hub.Module":
    def plus_one():
        x = tf.compat.v1.placeholder(dtype=tf.float32, name="x")
        y = x + 1
        hub.add_signature(inputs=x, outputs=y)

    spec = hub.create_module_spec(plus_one)
    with tf.compat.v1.get_default_graph().as_default():
        module = hub.Module(spec, trainable=True)
        return module


def test_import_from_tfhub(modelstore: "ModelStore"):
    identifier = _plus_one_model_tf1()
    tag = bentoml.tensorflow_v1.import_from_tfhub(
        identifier, "module_hub_tf1", model_store=modelstore
    )
    model = modelstore.get(tag)
    assert model.info.context["import_from_tfhub"]
    module = bentoml.tensorflow_v1.load(
        tag, tfhub_tags=[], load_as_wrapper=False, model_store=modelstore
    )
    assert module._is_hub_module_v1 is True
