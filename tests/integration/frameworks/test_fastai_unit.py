from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import PropertyMock

import torch
import pytest
import torch.functional as F
from fastai.data.core import TfmdDL
from fastai.data.core import Datasets
from fastai.data.core import DataLoaders
from fastai.test_utils import synth_learner
from fastai.torch_core import Module
from fastcore.foundation import L
from fastai.data.transforms import Transform

import bentoml
from bentoml.exceptions import BentoMLException
from tests.utils.frameworks.fastai_utils import custom_model
from tests.utils.frameworks.pytorch_utils import LinearModel

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture

learner = custom_model()


class _FakeLossFunc(Module):
    reduction = "none"

    def forward(self, x, y):
        return F.mse_loss(x, y)

    def activation(self, x):
        return x + 1

    def decodes(self, x):
        return 2 * x


class _Add1(Transform):
    def encodes(self, x):
        return x + 1

    def decodes(self, x):
        return x - 1


mock_learner = synth_learner(n_trn=5)
dl = TfmdDL(Datasets(torch.arange(50), tfms=[L(), [_Add1()]]))
mock_learner.dls = DataLoaders(dl, dl)
mock_learner.loss_func = _FakeLossFunc()


def test_raise_exceptions():
    with pytest.raises(BentoMLException) as excinfo:
        bentoml.fastai.save_model("invalid_learner", LinearModel())  # type: ignore (testing exception)
    assert "does not support saving pytorch" in str(excinfo.value)

    class ForbiddenType:
        pass

    with pytest.raises(
        BentoMLException,
        match=re.escape(
            f"'bentoml.fastai.save_model()' only support saving fastai 'Learner' object. Got {ForbiddenType.__class__.__name__} instead."
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
        runnable = model.to_runnable()
        runnable()

    assert "CUDA is available" in caplog.text
