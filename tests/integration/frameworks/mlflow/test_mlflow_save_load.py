import os

import mlflow
import mlflow.models
import numpy as np
import pytest

import bentoml.mlflow
from bentoml.exceptions import InvalidArgument
from tests.utils.frameworks.sklearn_utils import sklearn_model_data
from tests.utils.helpers import assert_have_file_extension

MODEL_NAME = __name__.split(".")[-1]

# fmt: off
res_arr = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2]
)
# fmt: on


def test_mlflow_save_load(modelstore):
    (model, data) = sklearn_model_data()
    tag = bentoml.mlflow.save(MODEL_NAME, model, mlflow.sklearn, model_store=modelstore)
    model_info = modelstore.get(tag)
    assert_have_file_extension(os.path.join(model_info.path, "saved_model"), ".pkl")

    loaded = bentoml.mlflow.load(tag, model_store=modelstore)
    np.testing.assert_array_equal(loaded.predict(data), res_arr)  # noqa


def test_invalid_mlflow_loader(modelstore):
    class Foo(mlflow.models.Model):
        pass

    with pytest.raises(InvalidArgument):
        bentoml.mlflow.save(MODEL_NAME, Foo, os, model_store=modelstore)
