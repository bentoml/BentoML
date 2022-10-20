from __future__ import annotations

import re
import typing as t
import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import PropertyMock

import torch
import pytest
import torch.nn as nn
import torch.functional as F
from fastai.data.core import TfmdDL
from fastai.data.core import Datasets
from fastai.data.core import DataLoaders
from fastai.test_utils import synth_learner
from fastai.torch_core import Module
from fastcore.foundation import L
from fastai.data.transforms import Transform

import bentoml
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from tests.integration.frameworks.models.fastai import custom_model

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture

learner = custom_model()


class LinearModel(nn.Module):
    def forward(self, x: t.Any) -> t.Any:
        return x


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


mock_learner = synth_learner(n_trn=5)
dl = TfmdDL(Datasets(torch.arange(50), tfms=[L(), [_Add1()]]))
mock_learner.dls = DataLoaders(dl, dl)
mock_learner.loss_func = _FakeLossFunc()


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


@patch("torch.load")
@patch("torch.cuda.current_device")
@patch("torch._C")
def test_cuda_available(
    mock_C: MagicMock,
    mock_current_device: MagicMock,
    mock_load: MagicMock,
    caplog: LogCaptureFixture,
):
    # mock_cuda_available.return_value = True
    mock_current_device.return_value = 0
    mock_load.return_value = mock_learner
    type(mock_C)._cuda_getDeviceCount = PropertyMock(return_value=Mock(return_value=1))
    type(mock_C)._cudart = PropertyMock(return_value=type)
    with caplog.at_level(logging.DEBUG):
        model = bentoml.fastai.save_model("tabular_learner", learner)
        _ = model.to_runnable()()

    assert "CUDA is available" in caplog.text


def test_batchable_exception():
    with pytest.raises(
        BentoMLException, match="Batchable signatures are not supported *"
    ):
        bentoml.fastai.save_model(
            "learner", mock_learner, signatures={"predict": {"batchable": True}}
        )


def test_raise_attribute_runnable_error():
    with pytest.raises(
        InvalidArgument, match="No method with name not_exist found for Learner of *"
    ):
        model = bentoml.fastai.save_model(
            "tabular_learner", learner, signatures={"not_exist": {"batchable": False}}
        )
        _ = model.to_runnable()()
