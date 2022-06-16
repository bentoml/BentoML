from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest

import bentoml
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.tensorflow_utils import CustomLayer
from tests.utils.frameworks.tensorflow_utils import custom_activation
from tests.utils.frameworks.tensorflow_utils import KerasSequentialModel

if TYPE_CHECKING:
    import tensorflow.keras as keras


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
    bento_model = bentoml.keras.save_model(MODEL_NAME, model, **kwargs)
    assert bento_model == bentoml.keras.get(bento_model.tag)
    assert_have_file_extension(bento_model.path, ".pb")
    loaded = bentoml.keras.load_model(bento_model.tag)
    predict_assert_equal(loaded)


@pytest.mark.parametrize("signatures", [None, {"__call__": {"batchable": True}}])
def test_keras_run(signatures) -> None:
    model = KerasSequentialModel()
    bento_model = bentoml.keras.save_model(MODEL_NAME, model, signatures=signatures)
    runner = bento_model.to_runner()
    runner.init_local()

    # keras model's `__call__` will return a Tensor instead of
    # ndarray, here we test if we convert the Tensor to ndarray
    run_res = runner.run([test_data])
    assert isinstance(run_res, np.ndarray)
    assert np.array_equal(run_res, res)
