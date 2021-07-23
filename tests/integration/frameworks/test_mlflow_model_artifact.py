import os

import mlflow
import numpy as np
import pytest

from bentoml._internal.exceptions import InvalidArgument
from bentoml.mlflow import MLflowModel
from tests._internal.frameworks.sklearn_utils import sklearn_model_data


def test_mlflow_save_load(tmpdir):
    (model, data) = sklearn_model_data()
    MLflowModel(model, mlflow.sklearn).save(tmpdir)

    # fmt: off
    assert os.path.exists(os.path.join(tmpdir, MLflowModel._MODEL_NAMESPACE, "model.pkl"))  # noqa

    mlflow_loaded = MLflowModel.load(tmpdir)
    np.testing.assert_array_equal(model.predict(data), mlflow_loaded.predict(data))  # noqa
    # fmt: on


def test_invalid_mlflow_loader(tmpdir):
    class Foo:
        pass

    with pytest.raises(InvalidArgument):
        MLflowModel(Foo, os).save(tmpdir)
