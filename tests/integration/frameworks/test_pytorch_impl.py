import math

import numpy as np
import torch
import pandas as pd
import pytest
import torch.nn as nn

import bentoml
from bentoml._internal.frameworks.pytorch import PyTorchTensorContainer
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.pytorch_utils import test_df
from tests.utils.frameworks.pytorch_utils import LinearModel
from tests.utils.frameworks.pytorch_utils import LinearModelWithBatchAxis
from tests.utils.frameworks.pytorch_utils import ExtendedModel
from tests.utils.frameworks.pytorch_utils import predict_df


@pytest.fixture(scope="module")
def models():
    def _():
        model: nn.Module = LinearModel()
        tag = bentoml.pytorch.save("pytorch_test", model)
        return tag

    return _


def test_pytorch_save_load(models):
    tag = models()
    assert_have_file_extension(bentoml.models.get(tag).path, ".pt")

    pytorch_loaded: nn.Module = bentoml.pytorch.load(tag)
    assert predict_df(pytorch_loaded, test_df) == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cpu", "cuda", "cuda:0"])
def test_pytorch_save_load_across_devices(dev, models):
    def is_cuda(model):
        return next(model.parameters()).is_cuda

    tag = models()
    loaded: nn.Module = bentoml.pytorch.load(tag)
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
def test_pytorch_runner_setup_run_batch(input_data):
    model = LinearModel()
    tag = bentoml.pytorch.save("pytorch_test", model)
    runner = bentoml.pytorch.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch(input_data)
    assert res.unsqueeze(dim=0).item() == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cuda", "cuda:0"])
def test_pytorch_runner_setup_on_gpu(dev):
    model = LinearModel()
    tag = bentoml.pytorch.save("pytorch_test", model)
    runner = bentoml.pytorch.load_runner(tag)

    assert torch.cuda.device_count() == runner.num_replica


@pytest.mark.parametrize(
    "bias_pair",
    [(0.0, 1.0), (-0.212, 1.1392)],
)
def test_pytorch_runner_with_partial_kwargs(bias_pair):

    N, D_in, H, D_out = 64, 1000, 100, 1
    x = torch.randn(N, D_in)
    model = ExtendedModel(D_in, H, D_out)

    tag = bentoml.pytorch.save("pytorch_test_extended", model)
    bias1, bias2 = bias_pair
    runner1 = bentoml.pytorch.load_runner(tag, partial_kwargs=dict(bias=bias1))

    runner2 = bentoml.pytorch.load_runner(tag, partial_kwargs=dict(bias=bias2))

    res1 = runner1.run_batch(x)[0][0].item()
    res2 = runner2.run_batch(x)[0][0].item()

    # tensor to float may introduce larger errors, so we bump rel_tol
    # from 1e-9 to 1e-6 just in case
    assert math.isclose(res1 - res2, bias1 - bias2, rel_tol=1e-6)


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_pytorch_container(batch_axis):

    single_tensor = torch.arange(6).reshape(2, 3)
    singles = [single_tensor, single_tensor + 1]
    batch_tensor = torch.stack(singles, dim=batch_axis)

    assert (
        PyTorchTensorContainer.singles_to_batch(singles, batch_axis=batch_axis)
        == batch_tensor
    ).all()
    assert (
        PyTorchTensorContainer.batch_to_singles(batch_tensor, batch_axis=batch_axis)[0]
        == single_tensor
    ).all()

    model = LinearModelWithBatchAxis()
    tag = bentoml.pytorch.save("pytorch_test_container", model)
    runner = bentoml.pytorch.load_runner(
        tag,
        partial_kwargs=dict(batch_axis=batch_axis),
    )

    single_tensor = torch.arange(5, dtype=torch.float32)
    singles = [single_tensor, single_tensor]
    batch_tensor = torch.stack(singles, dim=batch_axis)
    assert runner.run_batch(batch_tensor)[0][0] == 10.0
    assert runner.run(single_tensor)[0] == 10.0
