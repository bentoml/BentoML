import os
import pickle
from pathlib import Path

import mlflow
import pytest
import yaml

import bentoml.mlflow
import bentoml.sklearn
from bentoml.exceptions import BentoMLException
from tests.integration.frameworks.mlflow.test_mlflow_save_load import res_arr
from tests.utils.frameworks.sklearn_utils import sklearn_model_data

MODEL_NAME = __name__.split(".")[-1]

current_file = Path(__file__).parent


@pytest.fixture()
def pyfunc_tag(modelstore, tmpdir):
    def _(flavor):
        model, _ = sklearn_model_data()
        options = {"flavor": flavor}
        mlmodel = {
            "flavors": {
                "python_function": {
                    "env": "conda.yaml",
                    "loader_module": "mlflow.sklearn",
                    "model_path": "model.pkl",
                    "python_version": "3.8.8",
                }
            },
            "utc_time_created": "2021-10-16 22:39:02.952912",
        }
        with modelstore.register(
            MODEL_NAME,
            module=bentoml.mlflow.__name__,
            options=options,
            framework_context={"mlflow": mlflow.__version__},
        ) as ctx:
            os.makedirs(os.path.join(str(ctx.path), "saved_model"), exist_ok=True)
            _path = os.path.join(str(ctx.path), "saved_model", "model.pkl")
            with open(_path, "wb") as of:
                pickle.dump(model, of)
            with open(os.path.join(str(ctx.path), "saved_model", "MLmodel"), "w") as of:
                yaml.safe_dump(mlmodel, of)
            return ctx.tag

    return _


def test_mlflow_load_runner(modelstore):
    (model, data) = sklearn_model_data()
    tag = bentoml.mlflow.save(MODEL_NAME, model, mlflow.sklearn, model_store=modelstore)
    runner = bentoml.mlflow.load_runner(tag, model_store=modelstore)
    assert isinstance(runner, bentoml.sklearn._SklearnRunner)


def test_mlflow_pyfunc_runner(modelstore, pyfunc_tag):
    _, data = sklearn_model_data()
    tag = pyfunc_tag("mlflow.pyfunc")
    pyfunc_runner = bentoml.mlflow.load_runner(tag, model_store=modelstore)
    pyfunc_runner._setup()

    assert tag in pyfunc_runner.required_models
    assert pyfunc_runner.num_concurrency_per_replica == pyfunc_runner.num_replica == 1

    res = pyfunc_runner._run_batch(data)
    assert all(res == res_arr)


@pytest.mark.parametrize(
    "exc, ctag", [(BentoMLException, False), (BentoMLException, True)]
)
def test_mlflow_runner_exc(pyfunc_tag, modelstore, exc, ctag):
    with pytest.raises(exc):
        uri = str(Path(current_file, "SimpleMNIST").resolve())
        tag = pyfunc_tag("mlflow.nonexistent")
        if not ctag:
            tag = bentoml.mlflow.import_from_uri(uri, model_store=modelstore)
        _ = bentoml.mlflow.load_runner(tag, model_store=modelstore)


def test_mlflow_runner_forbidden_init():
    with pytest.raises(EnvironmentError):
        _ = bentoml.mlflow._MLflowRunner()
