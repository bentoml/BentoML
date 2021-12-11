import typing as t
from typing import TYPE_CHECKING

import numpy as np
import psutil
import pytest
import catboost as cbt
from catboost.core import CatBoost
from catboost.core import CatBoostRegressor
from catboost.core import CatBoostClassifier

import bentoml
import bentoml.models
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.sklearn_utils import test_df

if TYPE_CHECKING:
    from bentoml._internal.store import Tag
    from bentoml._internal.models import ModelStore


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


def save_procedure(
    model_params: t.Dict[str, t.Any],
    metadata: t.Dict[str, t.Any],
    _modelstore: "ModelStore",
) -> "Tag":
    catboost_model = create_catboost_model()
    tag_info = bentoml.catboost.save(
        "test_catboost_model",
        catboost_model,
        model_params=model_params,
        metadata=metadata,
        model_store=_modelstore,
    )
    return tag_info


def forbidden_procedure(_modelstore: "ModelStore"):
    catboost_model = create_catboost_model()
    with bentoml.models.create(
        "invalid_module",
        module=__name__,
        labels=None,
        options=None,
        context=None,
        metadata=None,
        _model_store=_modelstore,
    ) as ctx:
        catboost_model.save_model(ctx.path_of("saved_model.model"))
        return ctx.tag


@pytest.mark.parametrize(
    "model_params, metadata", [(dict(model_type="classifier"), {"acc": 0.876})]
)
def test_catboost_save_load(
    model_params: t.Dict[str, t.Any],
    metadata: t.Dict[str, t.Any],
    modelstore: "ModelStore",
) -> None:
    tag = save_procedure(model_params, metadata, _modelstore=modelstore)
    _model = bentoml.models.get(tag, _model_store=modelstore)
    assert _model.info.metadata is not None
    assert_have_file_extension(_model.path, ".cbm")

    cbt_loaded = bentoml.catboost.load(
        _model.tag, model_params=model_params, model_store=modelstore
    )
    assert isinstance(cbt_loaded, CatBoostClassifier)
    assert cbt_loaded.predict(test_df) == np.array([1])


def test_catboost_load_exc(modelstore: "ModelStore") -> None:
    tag = forbidden_procedure(_modelstore=modelstore)
    with pytest.raises(BentoMLException):
        _ = bentoml.catboost.load(tag, model_store=modelstore)


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
) -> None:
    model_params = {"model_type": model_type}
    info = save_procedure(model_params, {}, _modelstore=modelstore)
    cbt_loaded = bentoml.catboost.load(
        info, model_params=model_params, model_store=modelstore
    )

    assert isinstance(cbt_loaded, expected_model)


def test_catboost_runner_setup_run_batch(modelstore: "ModelStore") -> None:
    tag = save_procedure({}, {}, _modelstore=modelstore)
    runner = bentoml.catboost.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.num_concurrency_per_replica == 1
    assert runner.num_replica == psutil.cpu_count()
    assert runner.run_batch(test_df) == np.array([1])
