from pathlib import Path

import pytest
from sklearn.neighbors import KNeighborsClassifier

import bentoml.mlflow

current_file = Path(__file__).parent


@pytest.mark.parametrize(
    "uri",
    [
        Path(current_file, "SimpleMNIST").resolve(),
        Path(current_file, "NestedMNIST").resolve(),
    ],
)
def test_mlflow_import_from_uri(uri, modelstore):
    tag = bentoml.mlflow.import_from_uri(str(uri), model_store=modelstore)
    model_info = modelstore.get(tag)
    assert "flavor" in model_info.options

    run, uri = bentoml.mlflow.load_project(tag, model_store=modelstore)
    assert callable(run)


def test_mlflow_import_from_uri_mlmodel(modelstore):
    uri = Path(current_file, "sklearn_clf").resolve()
    tag = bentoml.mlflow.import_from_uri(str(uri), model_store=modelstore)
    model_info = modelstore.get(tag)
    assert "flavor" in model_info.options
    model = bentoml.mlflow.load(tag, model_store=modelstore)
    assert isinstance(model, KNeighborsClassifier)
