import os
from pathlib import Path

import numpy as np
import psutil
import pytest
import mlflow.sklearn

import bentoml
from bentoml.exceptions import BentoMLException
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.sklearn_utils import sklearn_model_data

current_file = Path(__file__).parent

MODEL_NAME = __name__.split(".")[-1]

# fmt: off
res_arr = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2]
)
# fmt: on

# MLFlow db initialization spews SQLAlchemy deprecation warnings
@pytest.mark.filterwarnings("ignore:.*:sqlalchemy.exc.SADeprecationWarning")
def test_mlflow_save_load():
    (model, data) = sklearn_model_data()
    uri = Path(current_file, "sklearn_clf")
    tracking_db = Path(current_file, "mlruns.db")
    if not uri.exists():
        mlflow.sklearn.save_model(model, uri.resolve())
    mlflow.set_tracking_uri("sqlite:///" + str(tracking_db))
    client = mlflow.tracking.MlflowClient()
    v = mlflow.register_model(str(uri), "sklearn_clf")
    client.transition_model_version_stage(
        name="sklearn_clf", version=v.version, stage="Staging"
    )

    bento_model = bentoml.mlflow.import_model(MODEL_NAME, str(uri.resolve()))
    model_info = bentoml.models.get(bento_model.tag)

    loaded = bentoml.mlflow.load_model(model_info.tag)
    np.testing.assert_array_equal(loaded.predict(data), res_arr)  # noqa


@pytest.fixture()
def invalid_save_with_no_mlmodel():
    uri = Path(current_file, "sklearn_clf").resolve()
    bento_model = bentoml.mlflow.import_model("sklearn_clf", str(uri))
    info = bentoml.models.get(bento_model.tag)
    os.remove(str(Path(info.path, "mlflow_model", "MLmodel").resolve()))
    return bento_model.tag


def test_invalid_load(invalid_save_with_no_mlmodel):
    with pytest.raises(FileNotFoundError):
        _ = bentoml.mlflow.load_model(invalid_save_with_no_mlmodel)


def test_mlflow_load_runner():
    (_, data) = sklearn_model_data()
    uri = Path(current_file, "sklearn_clf").resolve()
    bento_model = bentoml.mlflow.import_model(MODEL_NAME, str(uri))
    runner = bentoml.mlflow.get(bento_model.tag).to_runner()
    runner.init_local()

    assert bento_model.tag == runner.models[0].tag

    res = runner.predict.run(data)
    assert all(res == res_arr)


@pytest.mark.parametrize(
    "uri",
    [
        Path(current_file, "SimpleMNIST").resolve(),
        Path(current_file, "NestedMNIST").resolve(),
    ],
)
def test_mlflow_invalid_import_mlproject(uri):
    with pytest.raises(BentoMLException):
        _ = bentoml.mlflow.import_model(MODEL_NAME, str(uri))


def test_mlflow_import_models_url():
    tracking_db = Path(current_file, "mlruns.db")
    mlflow.set_tracking_uri("sqlite:///" + str(tracking_db))
    _ = bentoml.mlflow.import_model(MODEL_NAME, "models:/sklearn_clf/Staging")
