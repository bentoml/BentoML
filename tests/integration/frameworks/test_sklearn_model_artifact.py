import os
import typing as t

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from bentoml.sklearn import SklearnModel

test_df: "pd.DataFrame" = pd.DataFrame([[5.0, 4.0, 3.0, 1.0]])


def sklearn_model(num_data: t.Optional[int] = 4) -> "RandomForestClassifier":
    iris = load_iris()
    X, Y = iris.data[:, :num_data], iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    clr = RandomForestClassifier()
    clr.fit(X_train, Y_train)
    return clr


def test_sklearn_save_load(tmpdir):
    model = sklearn_model()
    SklearnModel(model).save(tmpdir)
    assert os.path.exists(SklearnModel.get_path(tmpdir, ".pkl"))

    sklearn_loaded: t.Any = SklearnModel.load(tmpdir)
    assert sklearn_loaded.predict(test_df)[0] == model.predict(test_df)[0]
