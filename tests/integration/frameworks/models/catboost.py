from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from sklearn import metrics
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    import bentoml._internal.external_typing as ext


framework = bentoml.catboost


def accurate_to(expected, accuracy: float) -> t.Callable[[ext.NpNdArray], bool]:
    def check(out):
        return metrics.accuracy_score(expected, out) >= accuracy

    return check


def generator_accurate_to(
    expected, accuracy: float
) -> t.Callable[[ext.NpNdArray], bool]:
    def check(out):
        score = metrics.accuracy_score(expected, next(out))
        return score >= accuracy

    return check


# Simulate data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7
)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


classification_model = FrameworkTestModel(
    name="classification",
    model=CatBoostClassifier().fit(X_train, y_train),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[X_test],
                        expected=accurate_to(y_test, 0.9466),
                    ),
                ],
                "staged_predict": [
                    Input(
                        input_args=[X_test],
                        expected=generator_accurate_to(y_test, 0.7),
                    ),
                ],
            },
        ),
    ],
)

models: list[FrameworkTestModel] = [classification_model]
