import os

import numpy as np
import pandas as pd
import xgboost as xgb

from bentoml.xgboost import XgBoostModel

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

test_df = pd.DataFrame([test_data])


def predict_df(model: xgb.core.Booster, df: pd.DataFrame):
    dm = xgb.DMatrix(df)
    res = model.predict(dm)
    return np.asarray([np.argmax(line) for line in res])


def xgboost_model():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # read in data
    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    dt = xgb.DMatrix(X_train, label=y_train)

    # specify parameters via map
    param = {'max_depth': 3, 'eta': 0.3, 'objective': 'multi:softprob', 'num_class': 2}
    bst = xgb.train(param, dt)

    return bst


def test_xgboost_save_load(tmpdir):
    model = xgboost_model()

    XgBoostModel(model).save(tmpdir)
    assert os.path.exists(XgBoostModel.get_path(tmpdir, '.model'))

    xg_loaded: xgb.core.Booster = XgBoostModel.load(tmpdir)
    assert predict_df(xg_loaded, test_df) == 1
