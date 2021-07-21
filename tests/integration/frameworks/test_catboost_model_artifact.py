import os

import pytest
from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor

from bentoml._internal.exceptions import InvalidArgument
from bentoml.catboost import CatBoostModel

# This can be used in a CatBoostService
test_data = {
    "mean radius": 10.80,
    "mean texture": 21.98,
    "mean perimeter": 68.79,
    "mean area": 359.9,
    "mean smoothness": 0.08801,
    "mean compactness": 0.05743,
    "mean concavity": 0.03614,
    "mean concave points": 0.2016,
    "mean symmetry": 0.05977,
    "mean fractal dimension": 0.3077,
    "radius error": 1.621,
    "texture error": 2.240,
    "perimeter error": 20.20,
    "area error": 20.02,
    "smoothness error": 0.006543,
    "compactness error": 0.02148,
    "concavity error": 0.02991,
    "concave points error": 0.01045,
    "symmetry error": 0.01844,
    "fractal dimension error": 0.002690,
    "worst radius": 12.76,
    "worst texture": 32.04,
    "worst perimeter": 83.69,
    "worst area": 489.5,
    "worst smoothness": 0.1303,
    "worst compactness": 0.1696,
    "worst concavity": 0.1927,
    "worst concave points": 0.07485,
    "worst symmetry": 0.2965,
    "worst fractal dimension": 0.07662,
}


@pytest.fixture(scope='session')
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


def test_cat_boost_save(tmpdir, CancerClassifier):
    model = CancerClassifier(tmpdir)
    CatBoostModel(model).save(tmpdir)
    assert os.path.exists(CatBoostModel.get_path(tmpdir, ".cbm"))

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
