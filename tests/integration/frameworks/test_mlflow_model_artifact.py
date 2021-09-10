import os

import mlflow
import numpy as np
import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.mlflow import MLflowModel
from tests._internal.frameworks.sklearn_utils import sklearn_model_data
from tests._internal.helpers import assert_have_file_extension


def test_mlflow_save_load(tmpdir):
    (model, data) = sklearn_model_data()
    MLflowModel(model, mlflow.sklearn).save(tmpdir)

    # fmt: off
    assert_have_file_extension(os.path.join(tmpdir, 'bentoml_saved_model'), '.pkl')

    mlflow_loaded = MLflowModel.load(tmpdir)
    np.testing.assert_array_equal(model.predict(data), mlflow_loaded.predict(data))  # noqa
    # fmt: on


def test_invalid_mlflow_loader(tmpdir):
    class Foo:
        pass

    with pytest.raises(InvalidArgument):
        MLflowModel(Foo, os).save(tmpdir)
