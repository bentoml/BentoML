import math

import numpy as np
import torch
import pandas as pd
import pytest
import torch.nn as nn

import bentoml
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.pytorch_utils import test_df
from tests.utils.frameworks.pytorch_utils import predict_df
from tests.utils.frameworks.pytorch_utils import LinearModel


@pytest.fixture(scope="module")
def models():
    def _(test_type):
        _model: nn.Module = LinearModel()
        if "trace" in test_type:
            tracing_inp = torch.ones(5)
            model = torch.jit.trace(_model, tracing_inp)
        else:
            model = torch.jit.script(_model)
        tag = bentoml.torchscript.save("torchscript_test", model)
        return tag

    return _


@pytest.mark.parametrize("test_type", ["tracedmodel", "scriptedmodel"])
def test_torchscript_save_load(test_type, models):
    tag = models(test_type)
    assert_have_file_extension(bentoml.models.get(tag).path, ".pt")

    torchscript_loaded: nn.Module = bentoml.torchscript.load(tag)
    assert predict_df(torchscript_loaded, test_df) == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cpu", "cuda", "cuda:0"])
@pytest.mark.parametrize("test_type", ["tracedmodel", "scriptedmodel"])
def test_torchscript_save_load_across_devices(dev, test_type, models):
    def is_cuda(model):
        return next(model.parameters()).is_cuda

    tag = models(test_type)
    loaded = bentoml.torchscript.load(tag)
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
@pytest.mark.parametrize("test_type", ["tracedmodel", "scriptedmodel"])
def test_torchscript_runner_setup_run_batch(input_data, models, test_type):
    tag = models(test_type)
    runner = bentoml.torchscript.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1

    res = runner.run_batch(input_data)
    assert res.unsqueeze(dim=0).item() == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cuda", "cuda:0"])
def test_torchscript_runner_setup_on_gpu(dev):
    tag = models(test_type)
    runner = bentoml.torchscript.load_runner(tag)

    assert torch.cuda.device_count() == runner.num_replica
