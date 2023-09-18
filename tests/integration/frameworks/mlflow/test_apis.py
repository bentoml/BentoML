from __future__ import annotations

import os
import typing as t
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import mlflow
import pytest
import mlflow.models
import mlflow.sklearn
import mlflow.tracking
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

import bentoml
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml._internal.models.model import ModelContext

if TYPE_CHECKING:
    from sklearn.utils import Bunch

    from bentoml import Tag
    from bentoml._internal import external_typing as ext

MODEL_NAME = __name__.split(".")[-1]

# fmt: off
res = np.array(
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

iris: Bunch = t.cast("Bunch", load_iris())
X: ext.NpNDArray = iris.data[:, :4]
Y: ext.NpNDArray = iris.target


@pytest.fixture(name="URI")
def iris_clf_model(tmp_path: Path) -> Path:
    URI = tmp_path / "IrisClf"
    model = KNeighborsClassifier()
    model.fit(X, Y)
    mlflow.sklearn.save_model(model, URI.resolve())
    return URI


# MLFlow db initialization spews SQLAlchemy deprecation warnings
@pytest.mark.filterwarnings("ignore:.*:sqlalchemy.exc.SADeprecationWarning")
def test_mlflow_save_load(URI: Path, tmp_path: Path):
    tracking_db = tmp_path / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(str(URI), "IrisClf")
    client.transition_model_version_stage(
        name="IrisClf",
        version=mv.version,
        stage="Staging",
    )
    bento_model = bentoml.mlflow.import_model(MODEL_NAME, str(URI.resolve()))
    # make sure the model can be imported with models:/
    model_uri = bentoml.mlflow.import_model(MODEL_NAME, "models:/IrisClf/Staging")

    pyfunc = bentoml.mlflow.load_model(bento_model.tag)
    np.testing.assert_array_equal(pyfunc.predict(X), res)
    np.testing.assert_array_equal(
        bentoml.mlflow.load_model(model_uri.tag).predict(X), res
    )


def test_wrong_module_load():
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        context=ModelContext("wrong_module", {"wrong_module": "1.0.0"}),
        signatures={},
    ) as ctx:
        tag = ctx.tag
        model = ctx

    with pytest.raises(
        NotFound, match=f"Model {tag} was saved with module {__name__}, "
    ):
        bentoml.mlflow.get(tag)

    with pytest.raises(
        NotFound, match=f"Model {tag} was saved with module {__name__}, "
    ):
        bentoml.mlflow.load_model(tag)

    with pytest.raises(
        NotFound, match=f"Model {tag} was saved with module {__name__}, "
    ):
        bentoml.mlflow.load_model(model)


def test_invalid_import():
    uri = Path(__file__).parent / "NoPyfunc"
    with pytest.raises(
        BentoMLException,
        match="does not support the required python_function flavor",
    ):
        _ = bentoml.mlflow.import_model("NoPyfunc", str(uri.resolve()))


@pytest.fixture(name="no_mlmodel")
def fixture_no_mlmodel(URI: Path) -> Tag:
    bento_model = bentoml.mlflow.import_model("IrisClf", str(URI))
    info = bentoml.models.get(bento_model.tag)
    os.remove(str(Path(info.path, "mlflow_model", "MLmodel").resolve()))
    return bento_model.tag


def test_invalid_load(no_mlmodel: Tag):
    with pytest.raises(OSError):
        _ = bentoml.mlflow.load_model(no_mlmodel)


def test_invalid_signatures_model(URI: Path):
    with pytest.raises(
        BentoMLException,
        match="MLflow pyfunc model support only the `predict` method, *",
    ):
        _ = bentoml.mlflow.import_model(
            MODEL_NAME,
            str(URI),
            signatures={
                "asdf": {"batchable": True},
                "no_predict": {"batchable": False},
            },
        )


def test_mlflow_load_runner(URI: Path):
    bento_model = bentoml.mlflow.import_model(MODEL_NAME, str(URI))
    runner = bentoml.mlflow.get(bento_model.tag).to_runner()
    runner.init_local()

    assert bento_model.tag == runner.models[0].tag

    np.testing.assert_array_equal(runner.run(X), res)


def test_mlflow_invalid_import_mlproject():
    uri = Path(__file__).parent / "MNIST"
    with pytest.raises(BentoMLException):
        _ = bentoml.mlflow.import_model(MODEL_NAME, str(uri))


def test_get_mlflow_model(URI: Path):
    bento_model = bentoml.mlflow.import_model(MODEL_NAME, str(URI))
    mlflow_model = bentoml.mlflow.get_mlflow_model(bento_model.tag)
    assert isinstance(mlflow_model, mlflow.models.Model)
