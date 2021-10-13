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

native_data = [[1, 2, 3, 4, 5]]
native_tensor = tf.constant(native_data, dtype=tf.float64)

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor = tf.ragged.constant(ragged_data, dtype=tf.float64)


def predict__model(model, tensor):
    return model(tensor)


@pytest.mark.parametrize(
    "model_class, input_type, predict_fn",
    [
        (KerasSequentialModel(), native_tensor, predict__model),
        (NativeModel(), native_tensor, predict__model),
        (NativeRaggedModel(), ragged_tensor, predict__model),
    ],
)
def test_tensorflow_v2_save_load(model_class, input_type, predict_fn, modelstore):
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class, model_store=modelstore)
    model_info = modelstore.get(tag)
    assert_have_file_extension(model_info.path, ".pb")
    model = bentoml.tensorflow.load(MODEL_NAME, model_store=modelstore)
    assert all(predict_fn(model, input_type) == predict_fn(model_class, input_type))


def test_tensorflow_v2_setup_run_batch(modelstore):
    model_class = NativeModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.tensorflow.load_runner(tag, model_store=modelstore)
    runner._setup()

    assert tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1
    assert runner._run_batch(native_data).numpy() == np.array([[15.]])


@pytest.mark.gpus
def test_tensorflow_v2_setup_on_gpu(modelstore):
    model_class = NativeModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.tensorflow.load_runner(tag, model_store=modelstore, resource_quota=dict(gpus=0))
    runner._setup()

    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == len(tf.config.list_physical_devices("GPU"))
    assert runner._run_batch(native_tensor) == np.array([[15.]])
