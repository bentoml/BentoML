import os

import pytest
from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor

from bentoml._internal.exceptions import InvalidArgument
from bentoml.catboost import CatBoostModel
from tests._internal.bento_services.catboost import CancerClassifier


def test_cat_boost_save(tmpdir):
    model = CancerClassifier(tmpdir)
    CatBoostModel(model).save(tmpdir)
    assert os.path.exists(
        CatBoostModel.get_path(tmpdir, CatBoostModel.CATBOOST_FILE_EXTENSION)
    )

    catboost_loaded = CatBoostModel.load(tmpdir)
    assert isinstance(catboost_loaded, CatBoost)


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
