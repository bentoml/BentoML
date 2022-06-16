import os

import numpy as np
import torch
import pytest
import torch.nn as nn

import bentoml
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.pytorch_utils import test_df
from tests.utils.frameworks.pytorch_utils import predict_df
from tests.utils.frameworks.pytorch_utils import LinearModel
from tests.utils.frameworks.pytorch_utils import (
    make_pytorch_lightning_linear_model_class,
)


@pytest.fixture(scope="module")
def models():
    def _(test_type, labels=None, custom_objects=None):
        _model: nn.Module = LinearModel()
        if "trace" in test_type:
            tracing_inp = torch.ones(5)
            model = torch.jit.trace(_model, tracing_inp)
            bento_model = bentoml.torchscript.save_model(
                "torchscript_test",
                model,
                labels=labels,
                custom_objects=custom_objects,
            )
        elif "script" in test_type:
            model = torch.jit.script(_model)
            bento_model = bentoml.torchscript.save_model(
                "torchscript_test",
                model,
                labels=labels,
                custom_objects=custom_objects,
            )
        elif "pytorch_lightning" in test_type:
            PlLinearModel = make_pytorch_lightning_linear_model_class()
            model = PlLinearModel()
            bento_model = bentoml.pytorch_lightning.save_model(
                "torchscript_test",
                model,
                labels=labels,
                custom_objects=custom_objects,
            )
        return bento_model

    return _


@pytest.mark.parametrize(
    "test_type",
    ["tracedmodel", "scriptedmodel", "pytorch_lightning"],
)
def test_torchscript_save_load(test_type, models):

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    bentomodel = models(test_type, labels=labels, custom_objects={"func": custom_f})
    assert_have_file_extension(bentomodel.path, ".pt")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    torchscript_loaded: nn.Module = bentoml.torchscript.load_model(bentomodel.tag)
    assert predict_df(torchscript_loaded, test_df) == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cpu", "cuda", "cuda:0"])
@pytest.mark.parametrize(
    "test_type",
    ["tracedmodel", "scriptedmodel", "pytorch_lightning"],
)
def test_torchscript_save_load_across_devices(dev, test_type, models):
    def is_cuda(model):
        return next(model.parameters()).is_cuda

    bento_model = models(test_type)
    loaded = bentoml.torchscript.load_model(bento_model.tag)
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
@pytest.mark.parametrize(
    "test_type",
    ["tracedmodel", "scriptedmodel", "pytorch_lightning"],
)
def test_torchscript_runner_setup_run_batch(input_data, models, test_type):
    bento_model = models(test_type)
    runner = bento_model.to_runner(cpu=4)

    assert bento_model.tag in [m.tag for m in runner.models]
    assert runner.scheduled_worker_count == 1

    runner.init_local()

    res = runner.run(input_data)
    assert res.unsqueeze(dim=0).item() == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("nvidia_gpu", [1, 2])
@pytest.mark.parametrize(
    "test_type",
    ["tracedmodel", "scriptedmodel", "pytorch_lightning"],
)
def test_torchscript_runner_setup_on_gpu(nvidia_gpu: int, models, test_type):
    bento_model = models(test_type)
    runner = bento_model.to_runner(nvidia_gpu=nvidia_gpu)
    assert runner.scheduled_worker_count == nvidia_gpu

    for i in range(nvidia_gpu):
        runner.setup_worker(i + 1)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == str(i)
