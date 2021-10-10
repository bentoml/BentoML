import json
import os
import typing as t

import numpy as np
import pandas as pd
import psutil
import pytest
import xgboost as xgb

import bentoml.xgboost
from bentoml.exceptions import BentoMLException
from tests._internal.frameworks.sklearn_utils import test_df
from tests._internal.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    from bentoml import ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


def predict_df(model: xgb.core.Booster, df: pd.DataFrame):
    dm = xgb.DMatrix(df)
    res = model.predict(dm)
    return np.asarray([np.argmax(line) for line in res])


def xgboost_model() -> "xgb.Booster":
    from sklearn.datasets import load_breast_cancer

    # read in data
    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    dt = xgb.DMatrix(X, label=y)

    # specify parameters via map
    param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}
    bst = xgb.train(param, dt)

    return bst


def wrong_module(modelstore: "ModelStore"):
    model = xgboost_model()
    with modelstore.register(
        "wrong_module",
        module=__name__,
        options=None,
        framework_context=None,
        metadata=None,
    ) as ctx:
        model.save_model(os.path.join(ctx.path, "saved_model.model"))
        return str(ctx.path)


@pytest.mark.parametrize(
    "booster_params, metadata",
    [
        (dict(), {"model": "Booster", "test": True}),
        (
            {"disable_default_eval_metric": 1, "nthread": 2, "tree_method": "hist"},
            {"acc": 0.876},
        ),
    ],
)
def test_xgboost_save_load(
    booster_params, metadata, modelstore
):  # noqa # pylint: disable
    model = xgboost_model()
    tag = bentoml.xgboost.save(
        TEST_MODEL_NAME,
        model,
        booster_params=booster_params,
        metadata=metadata,
        model_store=modelstore,
    )
    assert modelstore.get(tag).metadata is not None
    assert_have_file_extension(
        os.path.join(modelstore._base_dir, TEST_MODEL_NAME, tag.split(":")[-1]), ".json"
    )

    xgb_loaded = bentoml.xgboost.load(
        tag, model_store=modelstore, booster_params=booster_params
    )
    config = json.loads(xgb_loaded.save_config())
    if not booster_params:
        assert config["learner"]["generic_param"]["nthread"] == str(psutil.cpu_count())
    else:
        assert config["learner"]["generic_param"]["nthread"] == str(2)
    assert isinstance(xgb_loaded, xgb.Booster)
    assert predict_df(xgb_loaded, test_df) == 1


@pytest.mark.parametrize("exc", [BentoMLException])
def test_xgboost_load_exc(exc, modelstore):
    tag = wrong_module(modelstore)
    with pytest.raises(exc):
        bentoml.xgboost.load(tag, model_store=modelstore)


def test_xgboost_load_runner():
    ...
