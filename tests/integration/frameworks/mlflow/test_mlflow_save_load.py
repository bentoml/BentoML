import os
import pickle
from pathlib import Path

import mlflow
import mlflow.models
import numpy as np
import pytest

import bentoml.mlflow
from bentoml.exceptions import BentoMLException, InvalidArgument
from tests.utils.frameworks.sklearn_utils import sklearn_model_data
from tests.utils.helpers import assert_have_file_extension

current_file = Path(__file__).parent

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


@pytest.mark.parametrize("uri, expected", [("s3:/test", True), ("https://not", False)])
def test_is_s3_url(uri, expected):
    assert bentoml.mlflow._is_s3_url(uri) == expected


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


@pytest.fixture()
def invalid_save_with_no_mlmodel(modelstore):
    uri = Path(current_file, "sklearn_clf").resolve()
    tag = bentoml.mlflow.import_from_uri(
        "sklearn_clf", str(uri), model_store=modelstore
    )
    info = modelstore.get(tag)
    os.remove(str(Path(info.path, "sklearn_clf", "MLmodel").resolve()))
    return tag


def test_invalid_load(modelstore, invalid_save_with_no_mlmodel):
    with pytest.raises(BentoMLException):
        _ = bentoml.mlflow.load(invalid_save_with_no_mlmodel, model_store=modelstore)
