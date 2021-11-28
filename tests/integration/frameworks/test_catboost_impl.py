import typing as t
from typing import TYPE_CHECKING

import catboost as cbt
import numpy as np
import psutil
import pytest
from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor

import bentoml.catboost
from bentoml.exceptions import BentoMLException
from tests.utils.frameworks.sklearn_utils import test_df
from tests.utils.helpers import assert_have_file_extension
from tests.utils.types import InvalidModule, Pipeline

if TYPE_CHECKING:
    from bentoml._internal.models import Model, ModelStore


def create_catboost_model() -> cbt.core.CatBoostClassifier:
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    clf = CatBoostClassifier(
        iterations=2,
        depth=2,
        learning_rate=1,
        loss_function="Logloss",
        verbose=False,
    )

    # train the model
    clf.fit(X, y)

    return clf


def save_catboost_model(models: "Model") -> None:
    model = create_catboost_model()
    model.save_model(models.path_of("saved_model.model"))


@pytest.mark.parametrize(
    "model_params, metadata", [(dict(model_type="classifier"), {"acc": 0.876})]
)
def test_catboost_save_load(
    model_params: t.Dict[str, t.Any],
    metadata: t.Dict[str, t.Any],
    modelstore: "ModelStore",
    pipeline: Pipeline,
) -> None:
    _model = pipeline(
        create_catboost_model,
        bentoml.catboost,
        model_params=model_params,
        metadata=metadata,
    )
    assert _model.info.metadata is not None
    assert_have_file_extension(_model.path, ".cbm")

    cbt_loaded = bentoml.catboost.load(
        _model.tag, model_params=model_params, model_store=modelstore
    )
    assert isinstance(cbt_loaded, CatBoostClassifier)
    assert cbt_loaded.predict(test_df) == np.array([1])


def test_catboost_load_exc(
    modelstore: "ModelStore", invalid_module: InvalidModule
) -> None:
    tag = invalid_module(save_catboost_model)
    with pytest.raises(BentoMLException):
        bentoml.catboost.load(tag, model_store=modelstore)


@pytest.mark.parametrize(
    "model_type, expected_model",
    [
        ("regressor", CatBoostRegressor),
        ("classifier", CatBoostClassifier),
        ("", CatBoost),
    ],
)
def test_catboost_model_type(
    model_type: str,
    expected_model: t.Union[CatBoost, CatBoostClassifier, CatBoostRegressor],
    modelstore: "ModelStore",
    pipeline: Pipeline,
) -> None:
    model_params = dict(model_type=model_type)
    info = pipeline(create_catboost_model, bentoml.catboost, model_params=model_params)
    cbt_loaded = bentoml.catboost.load(
        info.tag, model_params=model_params, model_store=modelstore
    )

    assert isinstance(cbt_loaded, expected_model)


def test_catboost_runner_setup_run_batch(
    modelstore: "ModelStore", pipeline: Pipeline
) -> None:
    _model = pipeline(create_catboost_model, bentoml.catboost)
    runner = bentoml.catboost.load_runner(_model.tag, model_store=modelstore)

    assert _model.tag in runner.required_models
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == psutil.cpu_count()
    assert runner.run_batch(test_df) == np.array([1])
