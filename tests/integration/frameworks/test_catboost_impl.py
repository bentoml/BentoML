import os
import typing as t

import catboost as cbt
import pytest
from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor

import bentoml.catboost
from bentoml.exceptions import BentoMLException
from tests.utils.frameworks.sklearn_utils import test_df
from tests.utils.helpers import assert_have_file_extension

if t.TYPE_CHECKING:
    from bentoml._internal.models.store import ModelInfo, ModelStore

TEST_MODEL_NAME = __name__.split(".")[-1]


def catboost_model() -> cbt.core.CatBoostClassifier:
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


@pytest.fixture(scope="module")
def save_proc(
    modelstore: "ModelStore",
) -> t.Callable[[t.Dict[str, t.Any], t.Dict[str, t.Any]], "ModelInfo"]:
    def _(model_params, metadata) -> "ModelInfo":
        model = catboost_model()
        tag = bentoml.catboost.save(
            TEST_MODEL_NAME,
            model,
            model_params=model_params,
            metadata=metadata,
            model_store=modelstore,
        )
        info = modelstore.get(tag)
        return info

    return _


def wrong_module(modelstore: "ModelStore"):
    model = catboost_model()
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
    "model_params, metadata", [(dict(model_type="classifier"), {"acc": 0.876})]
)
def test_catboost_save_load(model_params, metadata, modelstore, save_proc):

    model = catboost_model()
    info = save_proc(model_params, metadata)
    assert info.metadata is not None
    assert_have_file_extension(info.path, ".cbm")

    cbt_loaded = bentoml.catboost.load(
        info.tag, model_params=model_params, model_store=modelstore
    )
    assert isinstance(cbt_loaded, CatBoostClassifier)
    assert cbt_loaded.predict(test_df) == model.predict(test_df)


@pytest.mark.parametrize("exc", [BentoMLException])
def test_catboost_load_exc(exc, modelstore):
    tag = wrong_module(modelstore)
    with pytest.raises(exc):
        bentoml.catboost.load(tag, model_store=modelstore)


@pytest.mark.parametrize(
    "model_type, expected_model",
    [
        ("regressor", CatBoostRegressor),
        ("classifier", CatBoostClassifier),
        ("", CatBoost),
    ],
)
def test_catboost_model_type(model_type, expected_model, modelstore, save_proc):
    model_params = dict(model_type=model_type)
    info = save_proc(model_params, None)
    cbt_loaded = bentoml.catboost.load(
        info.tag, model_params=model_params, model_store=modelstore
    )

    assert isinstance(cbt_loaded, expected_model)
