from __future__ import annotations

import json
import typing as t

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.xgboost

# read in data
cancer = load_breast_cancer()  # type: ignore (incomplete sklearn types)
dt = xgb.DMatrix(cancer.data, label=cancer.target)  # type: ignore

# specify parameters via map
param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}


def check_model(bst: xgb.Booster | xgb.XGBModel, resources: dict[str, t.Any]):
    if not isinstance(bst, xgb.Booster):
        bst = bst.get_booster()
    config = json.loads(bst.save_config())
    if "nvidia.com/gpu" in resources and resources["nvidia.com/gpu"] > 0:
        assert config["learner"]["generic_param"]["nthread"] == str(1)
    elif "cpu" in resources and resources["cpu"] > 0:
        assert config["learner"]["generic_param"]["nthread"] == str(
            int(resources["cpu"])
        )


def close_to(expected):
    def check_output(out):
        return np.isclose(out, expected).all()

    return check_output


cancer_model = FrameworkTestModel(
    name="cancer",
    model=xgb.train(param, dt),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[np.array([cancer.data[0]])],
                        expected=close_to([[0.87606, 0.123939]]),
                        preprocess=xgb.DMatrix,
                    ),
                    Input(
                        input_args=[np.array([cancer.data[1]])],
                        expected=close_to([[0.97558, 0.0244234]]),
                        preprocess=xgb.DMatrix,
                    ),
                ],
            },
            check_model=check_model,
        ),
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[pd.DataFrame([cancer.data[0]])],
                        expected=close_to([[0.87606, 0.123939]]),
                        preprocess=xgb.DMatrix,
                    ),
                    Input(
                        input_args=[pd.DataFrame([cancer.data[1]])],
                        expected=close_to([[0.97558, 0.0244234]]),
                        preprocess=xgb.DMatrix,
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)


iris = load_iris()

xgbc = xgb.XGBClassifier()
xgbc.fit(iris.data, iris.target)


iris_model = FrameworkTestModel(
    name="iris",
    model=xgbc,
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[np.array([iris.data[0]])],
                        expected=[iris.target[0]],
                    ),
                    Input(
                        input_args=[np.array(iris.data[1:5])],
                        expected=iris.target[1:5],
                    ),
                ],
            },
            check_model=check_model,
        ),
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[pd.DataFrame([iris.data[0]])],
                        expected=[iris.target[0]],
                    ),
                    Input(
                        input_args=[pd.DataFrame(iris.data[1:5])],
                        expected=iris.target[1:5],
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)


models: list[FrameworkTestModel] = [cancer_model, iris_model]
