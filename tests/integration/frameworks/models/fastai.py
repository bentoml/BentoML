from __future__ import annotations

import json

import numpy as np
import torch
import pandas as pd
import sklearn.datasets as datasets
from fastai.metrics import accuracy
from fastai.tabular.all import tabular_learner
from fastai.tabular.all import TabularDataLoaders

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.fastai

# read in data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
y = pd.Series(iris.target, name="label")
dl = TabularDataLoaders.from_df(
    df=pd.concat([X, y], axis=1), cont_names=list(X.columns), y_names="label"
)
model = tabular_learner(dl, metrics=accuracy, layers=[3])
model.fit(1)


iris_model = FrameworkTestModel(
    name="iris",
    model=model,
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[np.array([iris.data[0]])],
                        expected=lambda out: np.isclose(
                            out, [[0.87606, 0.123939]]
                        ).all(),
                        preprocess=torch.tensor,
                    ),
                    Input(
                        input_args=[np.array([iris.data[1]])],
                        expected=lambda out: np.isclose(
                            out, [[0.97558, 0.0244234]]
                        ).all(),
                        preprocess=torch.tensor,
                    ),
                ],
            },
        ),
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[torch.tensor([iris.data[0]])],
                        expected=lambda out: np.isclose(
                            out, [[0.87606, 0.123939]]
                        ).all(),
                    ),
                    Input(
                        input_args=[torch.tensor([iris.data[1]])],
                        expected=lambda out: np.isclose(
                            out, [[0.97558, 0.0244234]]
                        ).all(),
                    ),
                ],
            },
        ),
    ],
)
models: list[FrameworkTestModel] = [iris_model]
