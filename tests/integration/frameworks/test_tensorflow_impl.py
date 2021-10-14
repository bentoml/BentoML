import tempfile

import numpy as np
import psutil
import pytest
import tensorflow as tf

import bentoml.tensorflow
from tests._internal.frameworks.tensorflow_utils import (
    KerasSequentialModel,
    NativeModel,
    NativeRaggedModel,
)
from tests._internal.helpers import assert_have_file_extension

MODEL_NAME = __name__.split(".")[-1]
TF2 = tf.__version__.startswith("2")

import tensorflow_hub as hub

test_data = [[1.1, 2.2]]
test_tensor = tf.constant(test_data)

native_data = [[1, 2, 3, 4, 5]]
native_tensor = tf.constant(native_data, dtype=tf.float64)

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


def _model_dunder_call(model, tensor):
    if not TF2:
        pred_func = model.signatures["serving_default"]
        return pred_func(tensor)["prediction"]
    return model(tensor)


@pytest.fixture(scope="session")
def tf1_model_path():
    # Function below builds model graph
    def cnn_model_fn():
        X = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32, name="X")

        # dense layer
        inter1 = tf.compat.v1.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
        p = tf.argmax(input=inter1, axis=1)

        # loss
        y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="y")
        loss = tf.losses.softmax_cross_entropy(y, inter1)

        # training operation
        train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        return {"p": p, "loss": loss, "train_op": train_op, "X": X, "y": y}

    tf.compat.v1.reset_default_graph()
    cnn_model = cnn_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(cnn_model["p"], {cnn_model["X"]: test_data})

            inputs = {"X": cnn_model["X"]}
            outputs = {"prediction": cnn_model["p"]}

            tf.compat.v1.saved_model.simple_save(
                sess, temp_dir, inputs=inputs, outputs=outputs
            )
        yield temp_dir


@pytest.mark.skipif(TF2, reason="Tests for Tensorflow 1.x")
def test_tensorflow_v1_save_load(tf1_model_path, modelstore):
    tag = bentoml.tensorflow.save(
        "tensorflow_test", tf1_model_path, model_store=modelstore
    )
    model_info = modelstore.get(tag)
    assert_have_file_extension(model_info.path, ".pb")
    tf1_loaded = bentoml.tensorflow.load("tensorflow_test", model_store=modelstore)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        prediction = _model_dunder_call(tf1_loaded, test_tensor)
        assert prediction.shape == (1,)


@pytest.mark.skipif(TF2, reason="Tests for Tensorflow 1.x")
def test_tensorflow_v1_setup_run_batch(tf1_model_path, modelstore):
    tag = bentoml.tensorflow.save(
        "tensorflow_test", tf1_model_path, model_store=modelstore
    )
    runner = bentoml.tensorflow.load_runner(tag, model_store=modelstore)
    runner._setup()

    res = runner._run_batch(test_tensor)
    assert res.shape == (1,)


@pytest.mark.parametrize(
    "model_class, input_type, predict_fn, ragged",
    [
        (KerasSequentialModel(), native_tensor, _model_dunder_call, False),
        (NativeModel(), native_tensor, _model_dunder_call, False),
        (NativeRaggedModel(), ragged_tensor, _model_dunder_call, True),
    ],
)
@pytest.mark.skipif(not TF2, reason="Tests for Tensorflow 2.x")
def test_tensorflow_v2_save_load(
    model_class, input_type, predict_fn, modelstore, ragged
):
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class, model_store=modelstore)
    model_info = modelstore.get(tag)
    assert_have_file_extension(model_info.path, ".pb")
    model = bentoml.tensorflow.load(MODEL_NAME, model_store=modelstore)
    output = predict_fn(model, input_type)
    if ragged:
        assert all(output.numpy() == np.array([[15.0]] * 3))
    else:
        assert all(output.numpy() == np.array([[15.0]]))


@pytest.mark.skipif(not TF2, reason="Tests for Tensorflow 2.x")
def test_tensorflow_v2_setup_run_batch(modelstore):
    model_class = NativeModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.tensorflow.load_runner(tag, model_store=modelstore)
    runner._setup()

    assert tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1
    assert runner._run_batch(native_data) == np.array([[15.0]])


@pytest.mark.gpus
@pytest.mark.skipif(not TF2, reason="Tests for Tensorflow 2.x")
def test_tensorflow_v2_setup_on_gpu(modelstore):
    model_class = NativeModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.tensorflow.load_runner(
        tag, model_store=modelstore, resource_quota=dict(gpus=0), device_id="GPU:0"
    )
    runner._setup()

    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == len(tf.config.list_physical_devices("GPU"))
    assert runner._run_batch(native_tensor) == np.array([[15.0]])


def _plus_one_model_tf2():
    obj = tf.train.Checkpoint()

    @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
    def plus_one(x):
        return x + 1

    obj.__call__ = plus_one
    return obj


def _plus_one_model_tf1():
    def plus_one():
        x = tf.compat.v1.placeholder(dtype=tf.float32, name="x")
        y = x + 1
        hub.add_signature(inputs=x, outputs=y)

    spec = hub.create_module_spec(plus_one)
    with tf.compat.v1.get_default_graph().as_default():
        module = hub.Module(spec, trainable=True)
        return module


@pytest.mark.parametrize(
    "identifier, name, tags, is_module_v1, wrapped",
    [
        (_plus_one_model_tf1(), "module_hub_tf1", [], True, False),
        (_plus_one_model_tf2(), "saved_model_tf2", ["serve"], False, False),
        (
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            None,
            None,
            False,
            True,
        ),
    ],
)
@pytest.mark.skipif(not TF2, reason="We can tests TF1 functionalities with TF2 compat")
def test_import_from_tfhub(modelstore, identifier, name, tags, is_module_v1, wrapped):
    import tensorflow_text as text  # noqa # pylint: disable

    tag = bentoml.tensorflow.import_from_tfhub(identifier, name, model_store=modelstore)
    model_info = modelstore.get(tag)
    assert "tensorflow_hub" in model_info.context
    module = bentoml.tensorflow.load(
        tag, tfhub_tags=tags, load_as_wrapper=wrapped, model_store=modelstore
    )
    assert module._is_hub_module_v1 == is_module_v1
