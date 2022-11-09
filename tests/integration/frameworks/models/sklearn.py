from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    from sklearn.utils import Bunch

    from bentoml._internal import external_typing as ext

iris = t.cast("Bunch", load_iris())
X: ext.NpNDArray = iris.data[:, :4]
y: ext.NpNDArray = iris.target

framework = bentoml.sklearn

# fmt: off
res = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
)
# fmt: on

random_forest_classifier = FrameworkTestModel(
    name="classification",
    save_kwargs={
        "signatures": {
            "predict": {"batchable": False},
        }
    },
    model=RandomForestClassifier().fit(X, y),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[X],
                        expected=res,
                    ),
                ],
            },
        ),
    ],
)

models: list[FrameworkTestModel] = [random_forest_classifier]
