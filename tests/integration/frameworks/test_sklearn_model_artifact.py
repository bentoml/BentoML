import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from bentoml.sklearn import SklearnModel
from tests._internal.frameworks.sklearn_utils import sklearn_model_data


def test_sklearn_save_load(tmpdir):
    (model, data) = sklearn_model_data(clf=RandomForestClassifier)
    SklearnModel(model).save(tmpdir)
    assert os.path.exists(SklearnModel.get_path(tmpdir, ".pkl"))

    sklearn_loaded = SklearnModel.load(tmpdir)
    # fmt: off
    np.testing.assert_array_equal(model.predict(data), sklearn_loaded.predict(data))  # noqa
    # fmt: on
