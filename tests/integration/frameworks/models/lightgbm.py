from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.lightgbm

backward_compatible = True

# read in data
cancer = load_breast_cancer()  # type: ignore (incomplete sklearn types)
dt = lgb.basic.Dataset(cancer.data, label=cancer.target)  # type: ignore

# specify parameters via map
param = {
    "num_leaves": 31,
    "eta": 0.1,
    "num_iterations": 20,
    "objective": "softmax",
    "num_class": 2,
}


cancer_model = FrameworkTestModel(
    name="cancer",
    model=lgb.train(param, dt),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[np.array([cancer.data[0]])],
                        expected=lambda out: np.isclose(
                            out, [[0.8152496, 0.1847504]]
                        ).all(),
                    ),
                    Input(
                        input_args=[np.array([cancer.data[1]])],
                        expected=lambda out: np.isclose(
                            out, [[0.92220132, 0.07779868]]
                        ).all(),
                    ),
                ],
            },
        ),
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[pd.DataFrame([cancer.data[0]])],
                        expected=lambda out: np.isclose(
                            out, [[0.8152496, 0.1847504]]
                        ).all(),
                    ),
                    Input(
                        input_args=[pd.DataFrame([cancer.data[1]])],
                        expected=lambda out: np.isclose(
                            out, [[0.92220132, 0.07779868]]
                        ).all(),
                    ),
                ],
            },
        ),
    ],
)
models: list[FrameworkTestModel] = [cancer_model]
