import os

import pytest
from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor

from bentoml._internal.exceptions import InvalidArgument
from bentoml.catboost import CatBoostModel
from tests._internal.sklearn_utils import test_df


@pytest.fixture()
def CancerClassifier(tmpdir):
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
        train_dir=f"{tmpdir}/catboost_info",
    )

    # train the model
    clf.fit(X, y)

    return clf


def test_catboost_save_load(tmpdir, CancerClassifier):
    model = CancerClassifier
    CatBoostModel(model).save(tmpdir)
    assert os.path.exists(
        CatBoostModel.get_path(tmpdir, CatBoostModel.CATBOOST_EXTENSION)
    )

    catboost_loaded = CatBoostModel.load(tmpdir)
    assert isinstance(catboost_loaded, CatBoost)
    assert catboost_loaded.predict(test_df) == model.predict(test_df)


def test_invalid_catboost_load(tmpdir):
    with pytest.raises(InvalidArgument):
        CatBoostModel.load(tmpdir)


@pytest.mark.parametrize(
    "model_type, expected_model",
    [
        ("regressor", CatBoostRegressor),
        ("classifier", CatBoostClassifier),
        ("", CatBoost),
    ],
)
def test_catboost_model_type(model_type, expected_model):
    catboost_inst = CatBoostModel(model_type=model_type)
    assert isinstance(catboost_inst._model, expected_model)
