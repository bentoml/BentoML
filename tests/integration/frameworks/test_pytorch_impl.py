import math

import numpy as np
import torch
import pandas as pd
import psutil
import pytest
import torch.nn as nn

import bentoml
from bentoml.pytorch import PytorchTensorContainer
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.pytorch_utils import test_df
from tests.utils.frameworks.pytorch_utils import LinearModel


def predict_df(model: nn.Module, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    return model(input_tensor).unsqueeze(dim=0).item()


class LinearModelWithBatchAxis(nn.Module):
    def __init__(self):
        super(LinearModelWithBatchAxis, self).__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, x, batch_axis=0):
        if batch_axis == 1:
            x = x.permute([1, 0])
        res = self.linear(x)
        if batch_axis == 1:
            res = res.permute([0, 1])

        return res


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


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cpu", "cuda", "cuda:0"])
@pytest.mark.parametrize("test_type", ["", "tracedmodel", "scriptedmodel"])
def test_pytorch_save_load_across_devices(dev, test_type, modelstore, models):
    def is_cuda(model):
        return next(model.parameters()).is_cuda

    tag = models(test_type)
    loaded: nn.Module = bentoml.pytorch.load(tag, model_store=modelstore, device_id=dev)
    if dev == "cpu":
        assert not is_cuda(loaded)
    else:
        assert is_cuda(loaded)


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


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_pytorch_container(modelstore, batch_axis):

    single_tensor = torch.arange(6).reshape(2, 3)
    singles = [single_tensor, single_tensor + 1]
    batch_tensor = torch.stack(singles, dim=batch_axis)

    assert (
        PytorchTensorContainer.singles_to_batch(singles, batch_axis=batch_axis)
        == batch_tensor
    ).all()
    assert (
        PytorchTensorContainer.batch_to_singles(batch_tensor, batch_axis=batch_axis)[0]
        == single_tensor
    ).all()

    model = LinearModelWithBatchAxis()
    tag = bentoml.pytorch.save("pytorch_test_container", model, model_store=modelstore)
    batch_options = {
        "input_batch_axis": batch_axis,
        "output_batch_axis": batch_axis,
    }
    runner = bentoml.pytorch.load_runner(
        tag,
        model_store=modelstore,
        batch_options=batch_options,
        partial_kwargs=dict(batch_axis=batch_axis),
    )

    single_tensor = torch.arange(5, dtype=torch.float32)
    singles = [single_tensor, single_tensor]
    batch_tensor = torch.stack(singles, dim=batch_axis)
    assert runner.run_batch(batch_tensor)[0][0] == 10.0
    assert runner.run(single_tensor)[0] == 10.0
