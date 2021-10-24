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
from tests.utils.frameworks.sklearn_utils import test_df
from tests.utils.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    from bentoml._internal.models.store import ModelInfo, ModelStore

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


@pytest.fixture(scope="module")
def save_proc(
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(booster_params, metadata) -> "ModelInfo":
        model = xgboost_model()
        tag = bentoml.xgboost.save(
            TEST_MODEL_NAME,
            model,
            booster_params=booster_params,
            metadata=metadata,
            model_store=modelstore,
        )
        info = modelstore.get(tag)
        return info

    return _


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
    booster_params, metadata, modelstore, save_proc
):  # noqa # pylint: disable
    info = save_proc(booster_params, metadata)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".json")

    xgb_loaded = bentoml.xgboost.load(
        info.tag, model_store=modelstore, booster_params=booster_params
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


def test_xgboost_runner_setup_run_batch(modelstore, save_proc):
    booster_params = dict()
    info = save_proc(booster_params, None)
    runner = bentoml.xgboost.load_runner(info.tag, model_store=modelstore)

    assert info.tag in runner.required_models
    assert runner.num_concurrency_per_replica == psutil.cpu_count()
    assert runner.num_replica == 1

    assert np.asarray([np.argmax(_l) for _l in runner.run_batch(test_df)]) == 1
    assert isinstance(runner._model, xgb.Booster)


@pytest.mark.gpus
def test_xgboost_runner_setup_on_gpu(modelstore, save_proc):
    booster_params = dict()
    info = save_proc(booster_params, None)
    resource_quota = dict(gpus=0, cpu=0.4)
    runner = bentoml.xgboost.load_runner(
        info.tag, model_store=modelstore, resource_quota=resource_quota
    )

    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == 1
