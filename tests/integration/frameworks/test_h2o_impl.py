import json
import typing as t

import h2o
import pandas as pd
import pytest
import h2o.model
import h2o.automl

import bentoml.h2o

H2O_PORT = 54323

TEST_MODEL_NAME = __name__.split(".")[-1]

TEST_DATA = {
    "TemperatureCelcius": {"0": 21.6},
    "ExhaustVacuumHg": {"0": 62.52},
    "AmbientPressureMillibar": {"0": 1017.23},
    "RelativeHumidity": {"0": 67.87},
}


def predict_dataframe(
    model: "h2o.model.model_base.ModelBase", df: "pd.DataFrame"
) -> t.Optional[str]:
    hf = h2o.H2OFrame(df)
    pred = model.predict(hf)
    return pred.as_data_frame().to_json(orient="records")


@pytest.fixture(scope="module")
def train_h2o_aml() -> h2o.automl.H2OAutoML:

    h2o.init(port=H2O_PORT)
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


@pytest.mark.parametrize("metadata", [{"acc": 0.876}])
def test_h2o_save_load(train_h2o_aml, metadata, modelstore):
    test_df: pd.DataFrame = pd.read_json(json.dumps(TEST_DATA))
    model = train_h2o_aml.leader

    tag = bentoml.h2o.save(
        TEST_MODEL_NAME, model, metadata=metadata, model_store=modelstore
    )

    h2o_loaded: h2o.model.model_base.ModelBase = bentoml.h2o.load(
        tag,
        model_store=modelstore,
        init_params=dict(port=H2O_PORT),
    )
    # fmt: off
    assert predict_dataframe(train_h2o_aml.leader, test_df) == predict_dataframe(h2o_loaded, test_df)  # noqa
    # fmt: on
