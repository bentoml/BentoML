from __future__ import annotations

import json

from fastai.text.all import *

import bentoml
from bentoml._internal.runner.resource import Resource

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.fastai

# read in data
cancer = load_breast_cancer()  # type: ignore (incomplete sklearn types)
dt = xgb.DMatrix(cancer.data, label=cancer.target)  # type: ignore

# specify parameters via map
param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}


def check_model(bst: xgb.Booster, resource: Resource):
    config = json.loads(bst.save_config())
    print(config)
    if resource.nvidia_gpu is not None and resource.nvidia_gpu > 0:
        assert config["learner"]["generic_param"]["nthread"] == str(1)
    elif resource.cpu is not None and resource.cpu > 0:
        assert config["learner"]["generic_param"]["nthread"] == str(int(resource.cpu))


cancer_model = FrameworkTestModel(
    name="cancer",
    model=xgb.train(param, dt),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[np.array([cancer.data[0]])],
                        expected=lambda out: np.isclose(
                            out, [[0.87606, 0.123939]]
                        ).all(),
                        preprocess=xgb.DMatrix,
                    ),
                    Input(
                        input_args=[np.array([cancer.data[1]])],
                        expected=lambda out: np.isclose(
                            out, [[0.97558, 0.0244234]]
                        ).all(),
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
                        expected=lambda out: np.isclose(
                            out, [[0.87606, 0.123939]]
                        ).all(),
                        preprocess=xgb.DMatrix,
                    ),
                    Input(
                        input_args=[pd.DataFrame([cancer.data[1]])],
                        expected=lambda out: np.isclose(
                            out, [[0.97558, 0.0244234]]
                        ).all(),
                        preprocess=xgb.DMatrix,
                    ),
                ],
            },
            check_model=check_model,
        ),
    ],
)
models: list[FrameworkTestModel] = [cancer_model]
