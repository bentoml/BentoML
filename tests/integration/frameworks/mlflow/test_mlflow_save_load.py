import os

import mlflow
import mlflow.models
import numpy as np
import pytest

import bentoml.mlflow
from bentoml.exceptions import InvalidArgument
from tests.utils.frameworks.sklearn_utils import res_arr, sklearn_model_data
from tests.utils.helpers import assert_have_file_extension

MODEL_NAME = __name__.split(".")[-1]


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
