from __future__ import annotations

import json
import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    from sklearn.utils import Bunch

    from bentoml._internal import external_typing as ext

framework = bentoml.xgboost

backward_compatible = True

# read in data
cancer: Bunch = t.cast("Bunch", load_breast_cancer())
cancer_data = t.cast("ext.NpNDArray", cancer.data)
cancer_target = t.cast("ext.NpNDArray", cancer.target)
dt = xgb.DMatrix(cancer_data, label=cancer_target)

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


def close_to(expected: t.Any):
    def check_output(out: ext.NpNDArray):
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
                        input_args=[np.array([cancer_data[0]])],
                        expected=close_to([[0.87606, 0.123939]]),
                    ),
                    Input(
                        input_args=[np.array([cancer_data[1]])],
                        expected=close_to([[0.97558, 0.0244234]]),
                    ),
                ],
            },
            check_model=check_model,
        ),
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[pd.DataFrame([cancer_data[0]])],
                        expected=close_to([[0.87606, 0.123939]]),
                    ),
                    Input(
                        input_args=[pd.DataFrame([cancer_data[1]])],
                        expected=close_to([[0.97558, 0.0244234]]),
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)


iris: Bunch = t.cast("Bunch", load_iris())
iris_data = t.cast("ext.NpNDArray", iris.data)
iris_target = t.cast("ext.NpNDArray", iris.target)

xgbc = xgb.XGBClassifier()
xgbc.fit(iris_data, iris_target)


iris_model = FrameworkTestModel(
    name="iris",
    model=xgbc,
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[np.array([iris_data[0]])],
                        expected=[iris_target[0]],
                    ),
                    Input(
                        input_args=[np.array(iris_data[1:5])],
                        expected=iris_target[1:5],
                    ),
                ],
            },
            check_model=check_model,
        ),
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[pd.DataFrame([iris_data[0]])],
                        expected=[iris_target[0]],
                    ),
                    Input(
                        input_args=[pd.DataFrame(iris_data[1:5])],
                        expected=iris_target[1:5],
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)


models: list[FrameworkTestModel] = [cancer_model, iris_model]
