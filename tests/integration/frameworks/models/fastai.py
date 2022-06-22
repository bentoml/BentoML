from __future__ import annotations

import os

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import torch
import pandas as pd
import sklearn.datasets as datasets
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.data.block import DataBlock
from fastai.torch_core import Module
from fastai.torch_core import set_seed
from fastai.tabular.all import tabular_learner
from fastai.tabular.all import TabularDataLoaders

import bentoml
from tests.utils.frameworks.pytorch_utils import LinearModel

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    from sklearn.utils import Bunch

    import bentoml._internal.external_typing as ext

framework = bentoml.fastai

set_seed(123, reproducible=True)

iris: Bunch = datasets.load_iris()
X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
y = pd.Series(iris.target, name="label")

# read in data
def tabular_model() -> Learner:
    dl = TabularDataLoaders.from_df(
        df=pd.concat([X, y], axis=1),
        cont_names=list(X.columns),
        y_names="label",
        num_workers=0,
    )
    model = tabular_learner(dl, metrics=accuracy, layers=[3])
    model.fit(1)
    return model


class Loss(Module):
    reduction = "none"

    def forward(self, x: Any, _y: Any):
        return x

    def activation(self, x: Any):
        return x

    def decodes(self, x: Any):
        return x


def get_items(_x: Any) -> ext.NpNDArray:
    return np.ones([5, 5], np.float32)


def custom_model():
    model = LinearModel()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)
    learner.fit(1)
    return learner


iris_model = FrameworkTestModel(
    name="iris",
    model=tabular_model(),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=[X.iloc[0]],
                        expected=lambda out: np.isclose(
                            out[2].numpy(), [-0.35807556]
                        ).all(),
                    ),
                ],
            },
        ),
    ],
)


linear_regression = FrameworkTestModel(
    name="iris",
    model=custom_model(),
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=list(
                            map(
                                lambda x: x.astype(np.float32),
                                [np.array([[1] * 5])],
                            )
                        ),
                        expected=lambda out: np.isclose(
                            out[-1].squeeze().item(), 5.0
                        ).all(),
                    ),
                ],
            },
        ),
    ],
)
models: list[FrameworkTestModel] = [iris_model, linear_regression]
