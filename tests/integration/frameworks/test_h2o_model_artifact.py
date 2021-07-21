import json
import os
import typing as t

import h2o
import h2o.automl
import h2o.model
import pandas as pd
import pytest

from bentoml.h2o import H2oModel


def predict_dataframe(
    model: "h2o.model.model_base.ModelBase", df: "pd.DataFrame"
) -> t.Optional[str]:
    hf = h2o.H2OFrame(df)
    pred = model.predict(hf)
    return pred.as_data_frame().to_json(orient='records')


test_data = {
    "TemperatureCelcius": {"0": 21.6},
    "ExhaustVacuumHg": {"0": 62.52},
    "AmbientPressureMillibar": {"0": 1017.23},
    "RelativeHumidity": {"0": 67.87},
}


@pytest.fixture(scope="module")
def train_h2o_aml() -> h2o.automl.H2OAutoML:

    h2o.init()
    h2o.no_progress()

    df = h2o.import_file(
        "https://github.com/yubozhao/bentoml-h2o-data-for-testing/raw/master/"
        "powerplant_output.csv"
    )
    splits = df.split_frame(ratios=[0.8], seed=1)
    train = splits[0]
    test = splits[1]

    aml = h2o.automl.H2OAutoML(
        max_runtime_secs=60, seed=1, project_name="powerplant_lb_frame"
    )
    aml.train(y="HourlyEnergyOutputMW", training_frame=train, leaderboard_frame=test)

    return aml


def test_h2o_save_load(train_h2o_aml, tmpdir):
    test_df: pd.DataFrame = pd.read_json(json.dumps(test_data))
    H2oModel(train_h2o_aml.leader).save(tmpdir)
    assert os.path.exists(H2oModel.get_path(tmpdir, ""))

    h2o_loaded: h2o.model.model_base.ModelBase = H2oModel.load(tmpdir)
    assert predict_dataframe(train_h2o_aml.leader, test_df) == predict_dataframe(
        h2o_loaded, test_df
    )
