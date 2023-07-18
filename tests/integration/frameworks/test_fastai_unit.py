from __future__ import annotations

import os
import re
import typing as t

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.data.block import DataBlock
from fastai.data.core import DataLoaders
from fastai.data.core import Datasets
from fastai.data.core import Module
from fastai.data.core import TfmdDL
from fastai.data.transforms import Transform
from fastai.learner import Learner
from fastai.test_utils import synth_learner
from fastcore.foundation import L

import bentoml
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import InvalidArgument

if t.TYPE_CHECKING:
    import bentoml._internal.external_typing as ext


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x: t.Any):
        return self.linear(x)


class _FakeLossFunc(Module):
    reduction = "none"

    def forward(self, x: t.Any, y: t.Any):
        return F.mse_loss(x, y)

    def activation(self, x: t.Any):
        return x + 1

    def decodes(self, x: t.Any):
        return 2 * x


class _Add1(Transform):
    def encodes(self, x: t.Any):
        return x + 1

    def decodes(self, x: t.Any):
        return x - 1


@pytest.fixture
def learner() -> Learner:
    class Loss(Module):
        reduction = "none"

        def forward(self, x: t.Any, _y: t.Any):
            return x

        def activation(self, x: t.Any):
            return x

        def decodes(self, x: t.Any):
            return x

        def get_items(_: t.Any) -> ext.NpNDArray:
            return np.ones([5, 5], np.float32)

    def get_items(_: t.Any) -> ext.NpNDArray:
        return np.ones([5, 5], np.float32)

    model = LinearModel()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)
    learner.fit(1)
    return learner


def test_raise_exceptions():
    with pytest.raises(BentoMLException) as exc:
        bentoml.fastai.save_model("invalid_learner", LinearModel())  # type: ignore (testing exception)
    assert "does not support saving pytorch" in str(exc.value)

    class ForbiddenType:
        pass

    with pytest.raises(
        BentoMLException,
        match=re.escape(
            "'bentoml.fastai.save_model()' only support saving fastai 'Learner' object. Got module instead."
        ),
    ):
        bentoml.fastai.save_model("invalid_learner", ForbiddenType)  # type: ignore (testing exception)


def test_batchable_exception():
    mock_learner = synth_learner(n_trn=5)
    dl = TfmdDL(Datasets(torch.arange(50), tfms=[L(), [_Add1()]]))
    mock_learner.dls = DataLoaders(dl, dl)
    mock_learner.loss_func = _FakeLossFunc()

    with pytest.raises(
        BentoMLException, match="Batchable signatures are not supported *"
    ):
        bentoml.fastai.save_model(
            "learner", mock_learner, signatures={"predict": {"batchable": True}}
        )


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is not None, reason="Only run locally")
def test_raise_attribute_runnable_error(learner: Learner):
    with pytest.raises(
        InvalidArgument, match="No method with name not_exist found for Learner of *"
    ):
        model = bentoml.fastai.save_model(
            "tabular_learner", learner, signatures={"not_exist": {"batchable": False}}
        )
        _ = model.to_runnable()()
