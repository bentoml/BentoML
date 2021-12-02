import numpy as np
import psutil
import pytest
import tensorflow as tf
import tensorflow.keras as keras

import bentoml.keras
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.tensorflow_utils import CustomLayer
from tests.utils.frameworks.tensorflow_utils import custom_activation
from tests.utils.frameworks.tensorflow_utils import KerasSequentialModel

TF2 = tf.__version__.startswith("2")
MODEL_NAME = __name__.split(".")[-1]

test_data = [1, 2, 3, 4, 5]
res = KerasSequentialModel().predict(np.array([test_data]))


def predict_assert_equal(model: keras.Model):
    t_data = np.array([test_data])
    assert model.predict(t_data) == res


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (KerasSequentialModel(), {"store_as_json": True, "custom_objects": None}),
        (KerasSequentialModel(), {"store_as_json": False, "custom_objects": None}),
        (
            KerasSequentialModel(),
            {
                "store_as_json": False,
                "custom_objects": {
                    "CustomLayer": CustomLayer,
                    "custom_activation": custom_activation,
                },
            },
        ),
    ],
)
def test_keras_save_load(model, kwargs, modelstore):
    tag = bentoml.keras.save(MODEL_NAME, model, **kwargs, model_store=modelstore)
    model_info = modelstore.get(tag)
    if kwargs["custom_objects"] is not None:
        assert_have_file_extension(model_info.path, ".pkl")
    if kwargs["store_as_json"]:
        assert_have_file_extension(model_info.path, ".json")
        assert_have_file_extension(model_info.path, ".hdf5")
    else:
        assert_have_file_extension(model_info.path, ".h5")
    if not TF2:
        # Initialize variables in the graph/model
        bentoml.keras._sess.run(tf.global_variables_initializer())
        with bentoml.keras._default_sess():
            loaded = bentoml.keras.load(tag, model_store=modelstore)
            predict_assert_equal(loaded)
    else:
        loaded = bentoml.keras.load(tag, model_store=modelstore)
        predict_assert_equal(loaded)


@pytest.mark.skipif(not TF2, reason="Tests for Tensorflow 2.x")
def test_keras_v2_setup_run_batch(modelstore):
    model_class = KerasSequentialModel()
    tag = bentoml.keras.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.keras.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1
    assert runner.run_batch([test_data]) == res


@pytest.mark.skipif(TF2, reason="Tests for Tensorflow 1.x")
def test_keras_v1_setup_run_batch(modelstore):
    model_class = KerasSequentialModel()
    tag = bentoml.keras.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.keras.load_runner(tag, model_store=modelstore)
    with runner._session.as_default():

        assert runner.run_batch([test_data]) == res


@pytest.mark.gpus
def test_tensorflow_v2_setup_on_gpu(modelstore):
    model_class = KerasSequentialModel()
    tag = bentoml.keras.save(MODEL_NAME, model_class, model_store=modelstore)
    runner = bentoml.keras.load_runner(tag, model_store=modelstore)

    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == len(tf.config.list_physical_devices("GPU"))
    assert runner.run_batch([test_data]) == res
