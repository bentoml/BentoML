import numpy as np
import pandas as pd
import xgboost as xgb

import bentoml.xgboost as bxgb
from tests._internal.frameworks.sklearn_utils import test_df
from tests._internal.helpers import assert_have_file_extension


def predict_df(model: xgb.core.Booster, df: pd.DataFrame):
    dm = xgb.DMatrix(df)
    res = model.predict(dm)
    return np.asarray([np.argmax(line) for line in res])


def xgboost_model():
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


def test_xgboost_save_load(tmpdir):
    model = xgboost_model()

    _XgBoostModel(model).save(tmpdir)
    assert_have_file_extension(tmpdir, ".json")

    xg_loaded: xgb.core.Booster = _XgBoostModel.load(tmpdir)
    assert predict_df(xg_loaded, test_df) == 1
