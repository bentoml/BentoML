import typing as t

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras

import bentoml
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.tensorflow_utils import CustomLayer
from tests.utils.frameworks.tensorflow_utils import custom_activation
from tests.utils.frameworks.tensorflow_utils import KerasSequentialModel

MODEL_NAME = __name__.split(".")[-1]

test_data = [1, 2, 3, 4, 5]
res = KerasSequentialModel().predict(np.array([test_data]))


def predict_assert_equal(model: "keras.Model") -> None:
    t_data = np.array([test_data])
    assert model.predict(t_data) == res


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (
            KerasSequentialModel(),
            {},
        ),
        (
            KerasSequentialModel(),
            {
                "custom_objects": {
                    "CustomLayer": CustomLayer,
                    "custom_activation": custom_activation,
                },
            },
        ),
    ],
)
def test_keras_save_load(
    model: "keras.Model",
    kwargs: t.Dict[str, t.Any],
) -> None:
    tag = bentoml.keras.save_model(MODEL_NAME, model, **kwargs)
    bento_model = bentoml.keras.get(tag)
    assert_have_file_extension(bento_model.path, ".pb")
    loaded = bentoml.keras.load_model(tag)
    predict_assert_equal(loaded)


@pytest.mark.parametrize("signatures", [None, {"__call__": {"batchable": True}}])
def test_keras_run(signatures) -> None:
    model = KerasSequentialModel()
    tag = bentoml.keras.save_model(MODEL_NAME, model, signatures=signatures)
    runner = bentoml.keras.get(tag).to_runner()
    runner.init_local()

    # keras model's `__call__` will return a Tensor instead of
    # ndarray, here we test if we convert the Tensor to ndarray
    run_res = runner.run([test_data])
    assert isinstance(run_res, np.ndarray)
    assert np.array_equal(run_res, res)
