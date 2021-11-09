import numpy as np
import pandas as pd
import psutil
import pytest
import torch
import torch.nn as nn

import bentoml.pytorch
from tests.utils.frameworks.pytorch_utils import LinearModel, test_df
from tests.utils.helpers import assert_have_file_extension


def predict_df(model: nn.Module, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    return model(input_tensor).unsqueeze(dim=0).item()


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
