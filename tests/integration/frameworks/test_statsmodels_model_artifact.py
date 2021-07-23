import os
import typing as t

import joblib  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults

from bentoml.statsmodels import StatsModel

test_df = pd.DataFrame([[0, 0, 1, 1]])


def predict_df(model: t.Any, df: pd.DataFrame):
    return model.predict(int(df.iat[0, 0]))


# exported from
#  https://colab.research.google.com/github/bentoml/gallery/blob/master/statsmodels_holt/bentoml_statsmodels.ipynb
@pytest.fixture(scope="session")
def holt_model() -> "HoltWintersResults":
    df: pd.DataFrame = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
    )

    # Taking a test-train split of 80 %
    train = df[0 : int(len(df) * 0.8)]
    test = df[int(len(df) * 0.8) :]

    # Pre-processing the  Month  field
    train.Timestamp = pd.to_datetime(train.Month, format="%m-%d")
    train.index = train.Timestamp
    test.Timestamp = pd.to_datetime(test.Month, format="%m-%d")
    test.index = test.Timestamp

    # fitting the model based on  optimal parameters
    return ExponentialSmoothing(
        np.asarray(train["Sales"]), seasonal_periods=7, trend="add", seasonal="add",
    ).fit()


def test_statsmodels_save_load(tmpdir, holt_model):
    StatsModel(holt_model).save(tmpdir)
    assert os.path.exists(StatsModel.get_path(tmpdir, ".pkl"))

    statsmodels_loaded = StatsModel.load(tmpdir)
    assert (
        predict_df(statsmodels_loaded, test_df)[1] == predict_df(holt_model, test_df)[1]
    )
