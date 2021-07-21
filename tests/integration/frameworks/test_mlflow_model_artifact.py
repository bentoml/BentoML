import os
from collections import namedtuple

import numpy as np
import pytest
import sklearn.neighbors as skn
from sklearn import datasets

from bentoml.mlflow import MLflowModel

ModelWithData = namedtuple("ModelWithData", ['model', 'data'])


@pytest.fixture(scope='session')
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target
    knn_model = skn.KNeighborsClassifier()
    knn_model.fit(X, Y)
    return ModelWithData(model=knn_model, data=X)


def test_mlflow_save_load(tmpdir, sklearn_model):
    knn_model = sklearn_model.model
    MLflowModel(knn_model, 'sklearn').save(tmpdir)
    assert os.path.exists(MLflowModel.walk_path(tmpdir, ".pkl"))

    mlflow_loaded: "skn.KNeighborsClassifier" = MLflowModel.load(tmpdir)
    np.testing.assert_array_equal(
        knn_model.predict(sklearn_model.data), mlflow_loaded.predict(sklearn_model.data)
    )
