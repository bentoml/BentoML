import math

import numpy as np
import torch
import pandas as pd
import psutil
import pytest
import torch.nn as nn

import bentoml.pytorch
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.pytorch_utils import test_df
from tests.utils.frameworks.pytorch_utils import LinearModel


def predict_df(model: nn.Module, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    return model(input_tensor).unsqueeze(dim=0).item()


class ExtendedModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ExtendedModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x, bias=0.0):
        """
        In the forward function we accept a Tensor of input data and an optional bias
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred + bias


@pytest.fixture(scope="module")
def models(modelstore):
    def _(test_type):
        _model: nn.Module = LinearModel()
        if "trace" in test_type:
            tracing_inp = torch.ones(5)
            model = torch.jit.trace(_model, tracing_inp)
        elif "script" in test_type:
            model = torch.jit.script(_model)
        else:
            model = _model
        tag = bentoml.pytorch.save("pytorch_test", model, model_store=modelstore)
        return tag

    return _


@pytest.mark.parametrize("test_type", ["", "tracedmodel", "scriptedmodel"])
def test_pytorch_save_load(test_type, modelstore, models):
    tag = models(test_type)
    assert_have_file_extension(modelstore.get(tag).path, ".pt")

    pytorch_loaded: nn.Module = bentoml.pytorch.load(tag, model_store=modelstore)
    assert predict_df(pytorch_loaded, test_df) == 5.0


@pytest.mark.parametrize(
    "input_data",
    [
        test_df.to_numpy().astype(np.float32),
        torch.from_numpy(test_df.to_numpy().astype(np.float32)),
    ],
)
def test_pytorch_runner_setup_run_batch(modelstore, input_data):
    model = LinearModel()
    tag = bentoml.pytorch.save("pytorch_test", model, model_store=modelstore)
    runner = bentoml.pytorch.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert runner.num_concurrency_per_replica == psutil.cpu_count()

    res = runner.run_batch(input_data)
    assert res.unsqueeze(dim=0).item() == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cuda", "cuda:0"])
def test_pytorch_runner_setup_on_gpu(modelstore, dev):
    model = LinearModel()
    tag = bentoml.pytorch.save("pytorch_test", model, model_store=modelstore)
    runner = bentoml.pytorch.load_runner(tag, model_store=modelstore, device_id=dev)

    assert runner.num_concurrency_per_replica == 1
    assert torch.cuda.device_count() == runner.num_replica


@pytest.mark.parametrize(
    "bias_pair",
    [(0.0, 1.0), (-0.212, 1.1392)],
)
def test_pytorch_runner_with_partial_kwargs(modelstore, bias_pair):

    N, D_in, H, D_out = 64, 1000, 100, 1
    x = torch.randn(N, D_in)
    model = ExtendedModel(D_in, H, D_out)

    tag = bentoml.pytorch.save("pytorch_test_extended", model, model_store=modelstore)
    bias1, bias2 = bias_pair
    runner1 = bentoml.pytorch.load_runner(
        tag, model_store=modelstore, partial_kwargs=dict(bias=bias1)
    )

    runner2 = bentoml.pytorch.load_runner(
        tag, model_store=modelstore, partial_kwargs=dict(bias=bias2)
    )

    res1 = runner1.run_batch(x)[0][0].item()
    res2 = runner2.run_batch(x)[0][0].item()

    # tensor to float may introduce larger errors, so we bump rel_tol
    # from 1e-9 to 1e-6 just in case
    assert math.isclose(res1 - res2, bias1 - bias2, rel_tol=1e-6)
