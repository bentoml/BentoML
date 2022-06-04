import typing as t
from typing import TYPE_CHECKING

import numpy as np
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
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
) -> "Tag":
    catboost_model = create_catboost_model()
    tag_info = bentoml.catboost.save(
        "test_catboost_model",
        catboost_model,
        model_params=model_params,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    )
    return tag_info


def forbidden_procedure():
    catboost_model = create_catboost_model()
    with bentoml.models.create(
        "invalid_module",
        module=__name__,
        labels=None,
        options=None,
        context=None,
        metadata=None,
    ) as ctx:
        catboost_model.save_model(ctx.path_of("saved_model.model"))
        return ctx.tag


@pytest.mark.parametrize(
    "model_params, metadata",
    [
        (
            dict(model_type="classifier"),
            {"acc": 0.876},
        ),
    ],
)
def test_catboost_save_load(
    model_params: t.Dict[str, t.Any],
    metadata: t.Dict[str, t.Any],
) -> None:

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    tag = save_procedure(
        model_params,
        metadata,
        labels=labels,
        custom_objects={"func": custom_f},
    )
    _model = bentoml.models.get(tag)
    assert _model.info.metadata is not None
    assert_have_file_extension(_model.path, ".cbm")

    cbt_loaded = bentoml.catboost.load(_model.tag, model_params=model_params)
    assert isinstance(cbt_loaded, CatBoostClassifier)
    assert cbt_loaded.predict(test_df) == np.array([1])
    for k in labels.keys():
        assert labels[k] == _model.info.labels[k]
    assert _model.custom_objects["func"](3) == custom_f(3)


def test_catboost_load_exc() -> None:
    tag = forbidden_procedure()
    with pytest.raises(BentoMLException):
        _ = bentoml.catboost.load(tag)


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
) -> None:
    model_params = {"model_type": model_type}
    info = save_procedure(model_params, {})
    cbt_loaded = bentoml.catboost.load(info, model_params=model_params)

    assert isinstance(cbt_loaded, expected_model)


def test_catboost_runner_setup_run_batch() -> None:
    tag = save_procedure(
        {},
        {},
    )
    runner = bentoml.catboost.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert runner.run_batch(test_df) == np.array([1])
