import json

import pytest

import bentoml
from tests import (
    H2oExampleBentoService,
    build_api_server_docker_image,
    run_api_server_docker_container,
)

test_data = {
    "TemperatureCelcius": {"0": 21.6},
    "ExhaustVacuumHg": {"0": 62.52},
    "AmbientPressureMillibar": {"0": 1017.23},
    "RelativeHumidity": {"0": 67.87},
}


@pytest.fixture(scope="module")
def h2o_svc():
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init()

    df = h2o.import_file(
        "https://github.com/yubozhao/bentoml-h2o-data-for-testing/raw/master/"
        "powerplant_output.csv"
    )
    splits = df.split_frame(ratios=[0.8], seed=1)
    train = splits[0]
    test = splits[1]

    aml = H2OAutoML(max_runtime_secs=60, seed=1, project_name="powerplant_lb_frame")
    aml.train(y="HourlyEnergyOutputMW", training_frame=train, leaderboard_frame=test)

    svc = H2oExampleBentoService()
    svc.pack("model", aml.leader)

    return svc


@pytest.fixture(scope="module")
def h2o_svc_saved_dir(tmp_path_factory, h2o_svc):
    tmpdir = str(tmp_path_factory.mktemp("h2o_svc"))
    h2o_svc.save_to_dir(tmpdir)
    return tmpdir


@pytest.fixture()
def h2o_svc_loaded(h2o_svc_saved_dir):
    return bentoml.load(h2o_svc_saved_dir)


def test_h2o_artifact(h2o_svc_loaded):
    import pandas as pd

    test_df = pd.read_json(json.dumps(test_data))
    result = h2o_svc_loaded.predict(test_df)
    result_json = json.loads(result)
    inference_result = result_json[0]["predict"]
    assert (
        inference_result > 448 and inference_result < 453
    ), "Prediction on the saved h2o artifact does not match expect result"


@pytest.fixture()
def h2o_image(h2o_svc_saved_dir):
    with build_api_server_docker_image(
        h2o_svc_saved_dir, "h2o_example_service"
    ) as image:
        yield image


@pytest.fixture()
def h2o_docker_host(h2o_image):
    with run_api_server_docker_container(h2o_image, timeout=500) as host:
        yield host


@pytest.mark.skip(reason="Test currently failling on CI environment")
def test_h2o_artifact_with_docker(h2o_docker_host):
    import requests

    result = requests.post(
        f"http://{h2o_docker_host}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"},
    )
    assert result.status_code == 200, "Failed to make successful request"
    result_json = json.loads(result.json())
    inference_result = result_json[0]["predict"]
    assert (
        inference_result > 448 and inference_result < 453
    ), "Prediction on the saved h2o artifact does not match expect result"
