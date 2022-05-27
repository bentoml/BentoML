# import math

import numpy as np
import torch
import pytest
import torch.nn as nn

import bentoml
from tests.utils.helpers import assert_have_file_extension
from bentoml._internal.tag import Tag
from bentoml._internal.frameworks.pytorch import PyTorchTensorContainer

# from tests.utils.frameworks.pytorch_utils import ExtendedModel
from tests.utils.frameworks.pytorch_utils import test_df
from tests.utils.frameworks.pytorch_utils import predict_df
from tests.utils.frameworks.pytorch_utils import LinearModel

# TODO: signatures
# TODO: to_payload


@pytest.fixture(scope="module")
def model():
    return LinearModel()


@pytest.fixture(scope="module")
def model_tag(model: nn.Module):
    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    return bentoml.pytorch.save_model(
        "pytorch_test",
        model,
        labels=labels,
        custom_objects={"func": custom_f},
    )


def test_pytorch_save_load(model: nn.Module):
    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    tag = bentoml.pytorch.save_model(
        "pytorch_test",
        model,
        labels=labels,
        custom_objects={"func": custom_f},
    )
    bentomodel = bentoml.models.get(tag)
    assert_have_file_extension(bentomodel.path, ".pt")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    pytorch_loaded: nn.Module = bentoml.pytorch.load_model(tag)
    assert predict_df(pytorch_loaded, test_df) == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cpu", "cuda", "cuda:0"])
def test_pytorch_save_load_across_devices(dev: str, model_tag: Tag):
    def is_cuda(model: nn.Module) -> bool:
        return next(model.parameters()).is_cuda

    loaded: nn.Module = bentoml.pytorch.load_model(model_tag, device_id=dev)
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
def test_pytorch_runner_setup_run_batch(input_data, model_tag: Tag):
    runner = bentoml.pytorch.get(model_tag).to_runner(cpu=4)

    assert model_tag in [m.tag for m in runner.models]

    numprocesses = runner.scheduling_strategy.get_worker_count(
        runner.runnable_class, runner.resource_config
    )
    assert numprocesses == 1

    runner.init_local()
    res = runner.run(input_data)
    assert res.unsqueeze(dim=0).item() == 5.0


@pytest.mark.gpus
@pytest.mark.parametrize("nvidia_gpu", [1, 2])
def test_pytorch_runner_setup_on_gpu(nvidia_gpu: int, model_tag: Tag):
    runner = bentoml.pytorch.get(model_tag).to_runner(nvidia_gpu=nvidia_gpu)
    numprocesses = runner.scheduling_strategy.get_worker_count(
        runner.runnable_class, runner.resource_config
    )
    assert numprocesses == nvidia_gpu


"""
@pytest.mark.parametrize(
    "bias_pair",
    [(0.0, 1.0), (-0.212, 1.1392)],
)
def test_pytorch_runner_with_partial_kwargs(bias_pair):

    N, D_in, H, D_out = 64, 1000, 100, 1
    x = torch.randn(N, D_in)
    model = ExtendedModel(D_in, H, D_out)

    tag = bentoml.pytorch.save_model("pytorch_test_extended", model)
    bias1, bias2 = bias_pair
    runner1 = bentoml.pytorch.load_runner(tag, partial_kwargs=dict(bias=bias1))

    runner2 = bentoml.pytorch.load_runner(tag, partial_kwargs=dict(bias=bias2))

    res1 = runner1.run_batch(x)[0][0].item()
    res2 = runner2.run_batch(x)[0][0].item()

    # tensor to float may introduce larger errors, so we bump rel_tol
    # from 1e-9 to 1e-6 just in case
    assert math.isclose(res1 - res2, bias1 - bias2, rel_tol=1e-6)
"""


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_pytorch_container(batch_axis: int):
    one_batch = torch.arange(6).reshape(2, 3)
    batch_list = [one_batch, one_batch + 1]
    merged_batch = torch.cat(batch_list, dim=batch_axis)

    batches, indices = PyTorchTensorContainer.batches_to_batch(
        batch_list,
        batch_dim=batch_axis,
    )
    assert batches.shape == merged_batch.shape
    assert (batches == merged_batch).all()
    assert (
        PyTorchTensorContainer.batch_to_batches(
            merged_batch,
            indices=indices,
            batch_dim=batch_axis,
        )[0]
        == one_batch
    ).all()
