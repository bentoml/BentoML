import os
import typing as t

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import bentoml.sklearn
from bentoml.exceptions import BentoMLException
from tests._internal.frameworks.sklearn_utils import sklearn_model_data, test_df
from tests._internal.helpers import assert_have_file_extension

# fmt: off
test_res_array = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
)

# fmt: on

_MT = t.TypeVar("_MT")

if t.TYPE_CHECKING:
    from bentoml import ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


def wrong_module(modelstore: "ModelStore"):
    model, data = sklearn_model_data(clf=RandomForestClassifier)
    with modelstore.register(
        "wrong_module",
        module=__name__,
        options=None,
        metadata=None,
        framework_context=None,
    ) as ctx:
        joblib.dump(model, os.path.join(ctx.path, "saved_model.pkl"))
        return str(ctx.path)


@pytest.mark.parametrize(
    "metadata",
    [
        ({"model": "Sklearn", "test": True}),
        ({"acc": 0.876}),
    ],
)
def test_sklearn_save_load(metadata, modelstore):  # noqa # pylint: disable
    model, data = sklearn_model_data(clf=RandomForestClassifier)
    tag = bentoml.sklearn.save(
        TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
    )
    info = modelstore.get(tag)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".pkl")

    sklearn_loaded = bentoml.sklearn.load(tag, model_store=modelstore)

    assert isinstance(sklearn_loaded, RandomForestClassifier)

    np.testing.assert_array_equal(
        model.predict(data), sklearn_loaded.predict(data)
    )  # noqa

    np.testing.assert_array_equal(model.predict(data), test_res_array)


@pytest.mark.parametrize("exc", [BentoMLException])
def test_sklearn_load_exc(exc, modelstore):
    tag = wrong_module(modelstore)
    with pytest.raises(exc):
        bentoml.sklearn.load(tag, model_store=modelstore)


def test_sklearn_load_runner():
    ...
