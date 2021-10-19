from pathlib import Path

import pytest
from sklearn.neighbors import KNeighborsClassifier

import bentoml.mlflow
from bentoml.exceptions import BentoMLException

current_file = Path(__file__).parent

MODULE_NAME = __name__.split(".")[-1]


@pytest.mark.parametrize(
    "uri",
    [
        Path(current_file, "SimpleMNIST").resolve(),
        Path(current_file, "NestedMNIST").resolve(),
    ],
)
def test_mlflow_import_from_uri(uri, modelstore):
    tag = bentoml.mlflow.import_from_uri(MODULE_NAME, str(uri), model_store=modelstore)
    model_info = modelstore.get(tag)
    assert "flavor" in model_info.options

    with pytest.raises(BentoMLException):
        _ = bentoml.mlflow.load(tag, model_store=modelstore)


def test_mlflow_import_from_uri_mlmodel(modelstore):
    uri = Path(current_file, "sklearn_clf").resolve()
    tag = bentoml.mlflow.import_from_uri(
        "sklearn_clf", str(uri), model_store=modelstore
    )
    model_info = modelstore.get(tag)
    assert "flavor" in model_info.options
    model = bentoml.mlflow.load(tag, model_store=modelstore)
    assert isinstance(model, KNeighborsClassifier)
