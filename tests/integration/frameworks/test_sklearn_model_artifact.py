import numpy as np
from sklearn.ensemble import RandomForestClassifier

from bentoml.sklearn import SklearnModel
from tests._internal.frameworks.sklearn_utils import sklearn_model_data
from tests._internal.helpers import assert_have_file_extension


def test_sklearn_save_load(tmpdir):
    (model, data) = sklearn_model_data(clf=RandomForestClassifier)
    SklearnModel(model).save(tmpdir)
    assert_have_file_extension(tmpdir, ".pkl")

    sklearn_loaded = SklearnModel.load(tmpdir)
    # fmt: off
    np.testing.assert_array_equal(model.predict(data), sklearn_loaded.predict(data))  # noqa
    # fmt: on
