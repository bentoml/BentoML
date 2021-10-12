import json
import os
import typing as t

import numpy as np
import pandas as pd
import psutil
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier

import bentoml.sklearn
from bentoml.exceptions import BentoMLException
from tests._internal.frameworks.sklearn_utils import test_df
from tests._internal.helpers import assert_have_file_extension

_MT = t.TypeVar("_MT")

if t.TYPE_CHECKING:
    from bentoml import ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


def predict_df(model: _MT, df: pd.DataFrame):
    res = model.predict(df)
    return np.asarray([np.argmax(line) for line in res])


def sklearn_model() -> _MT:
    from sklearn.datasets import load_iris

    # read in data
    iris = load_iris()

    X = iris.data
    y = iris.target

    clf = svm.SVC(gamma="scale")
    clf.fit(X, y)

    return clf


def wrong_module(modelstore: "ModelStore"):
    model = sklearn_model()
    with modelstore.register(
        "wrong_module",
        module=__name__,
        metadata=None,
        framework_context=None,
    ) as ctx:
        joblib.dump(model, os.path.join(ctx.path, "saved_model.model"))
        return str(ctx.path)


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model": "Sklearn", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_sklearn_save_load(metadata, modelstore):  # noqa # pylint: disable
    model = sklearn_model()
    tag = bentoml.sklearn.save(
        TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
    )
    info = modelstore.get(tag)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".json")

    sklearn_loaded = bentoml.sklearn.load(tag, model_store=modelstore)

    assert isinstance(sklearn_loaded, _MT)
    assert predict_df(sklearn_loaded, test_df) == 1
    np.testing.assert_array_equal(
        model.predict(data), sklearn_loaded.predict(data)
    )  # noqa


@pytest.mark.parametrize("exc", [BentoMLException])
def test_sklearn_load_exc(exc, modelstore):
    tag = wrong_module(modelstore)
    with pytest.raises(exc):
        bentoml.sklearn.load(tag, model_store=modelstore)


def test_sklearn_load_runner():
    ...
