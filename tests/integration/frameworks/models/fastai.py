from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch.nn as nn
from fastai.learner import Learner
from fastai.metrics import accuracy
from sklearn.datasets import load_iris
from fastai.data.block import DataBlock
from fastai.torch_core import Module
from fastai.torch_core import set_seed
from fastai.tabular.all import tabular_learner
from fastai.tabular.all import TabularDataLoaders

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

if TYPE_CHECKING:
    from sklearn.utils import Bunch

    import bentoml._internal.external_typing as ext

framework = bentoml.fastai

SEED = 123

backward_compatible = False

set_seed(SEED, reproducible=True)

iris: Bunch = t.cast("Bunch", load_iris())
X = pd.DataFrame(
    t.cast("ext.NpNDArray", iris.data[:, :2]),
    columns=t.cast("list[str]", iris.feature_names[:2]),
)
y = pd.Series(t.cast("ext.NpNDArray", iris.target), name="label")


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x: t.Any):
        return self.linear(x)


class Loss(Module):
    reduction = "none"

    def forward(self, x: t.Any, _y: t.Any):
        return x

    def activation(self, x: t.Any):
        return x

    def decodes(self, x: t.Any):
        return x


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


def custom_model():
    def get_items(_: t.Any) -> ext.NpNDArray:
        return np.ones([5, 5], np.float32)

    model = LinearModel()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)
    learner.fit(1)
    return learner


def inputs(x: list[ext.NpNDArray]) -> list[ext.NpNDArray]:
    return list(map(lambda y: y.astype(np.float32), x))


def close_to(
    expected: float,
) -> t.Callable[[tuple[t.Any, t.Any, ext.NpNDArray]], np.bool_]:
    def check(out: tuple[t.Any, t.Any, ext.NpNDArray]) -> np.bool_:
        return np.isclose(out[-1].squeeze().item(), expected).all()

    return check


iris_model = FrameworkTestModel(
    name="iris",
    model=tabular_model(),
    model_signatures={"predict": {"batchable": False}},
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
    model_signatures={"predict": {"batchable": False}},
    configurations=[
        Config(
            test_inputs={
                "predict": [
                    Input(
                        input_args=inputs([np.array([[1] * 5])]),
                        expected=close_to(5.0),
                    ),
                ],
            },
        ),
    ],
)
models: list[FrameworkTestModel] = [iris_model, linear_regression]
