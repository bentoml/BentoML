import math

import pandas as pd
import psutil
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

import bentoml.pytorch_lightning
from tests.utils.helpers import assert_have_file_extension

test_df = pd.DataFrame([[5, 4, 3, 2]])


class AdditionModel(pl.LightningModule):
    def forward(self, inputs):
        return inputs.add(1)


class ExtendedModel(pl.LightningModule):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ExtendedModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x: torch.Tensor, bias: float = 0.0):
        """
        In the forward function we accept a Tensor of input data and an optional bias
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred + bias


def predict_df(model: pl.LightningModule, df: pd.DataFrame):
    input_tensor = torch.from_numpy(df.to_numpy())
    return model(input_tensor).numpy().tolist()


def test_pl_save_load(modelstore):
    model: "pl.LightningModule" = AdditionModel()
    tag = bentoml.pytorch_lightning.save(
        "pytorch_lightning_test", model, model_store=modelstore
    )
    info = modelstore.get(tag)
    assert_have_file_extension(info.path, ".pt")

    pl_loaded: "pl.LightningModule" = bentoml.pytorch_lightning.load(
        tag, model_store=modelstore
    )

    assert predict_df(pl_loaded, test_df) == [[6, 5, 4, 3]]


def test_pytorch_lightning_runner_setup_run_batch(modelstore):
    model: "pl.LightningModule" = AdditionModel()
    tag = bentoml.pytorch_lightning.save(
        "pytorch_lightning_test", model, model_store=modelstore
    )
    runner = bentoml.pytorch_lightning.load_runner(tag, model_store=modelstore)

    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert runner.num_concurrency_per_replica == psutil.cpu_count()

    res = runner.run_batch(torch.from_numpy(test_df.to_numpy()))
    assert res.numpy().tolist() == [[6, 5, 4, 3]]


@pytest.mark.gpus
@pytest.mark.parametrize("dev", ["cuda", "cuda:0"])
def test_pytorch_lightning_runner_setup_on_gpu(modelstore, dev):
    model: "pl.LightningModule" = AdditionModel()
    tag = bentoml.pytorch_lightning.save(
        "pytorch_lightning_test", model, model_store=modelstore
    )
    runner = bentoml.pytorch_lightning.load_runner(
        tag, model_store=modelstore, device_id=dev
    )

    assert runner.num_concurrency_per_replica == 1
    assert torch.cuda.device_count() == runner.num_replica


@pytest.mark.parametrize(
    "bias_pair",
    [(0.0, 1.0), (-0.212, 1.1392)],
)
def test_pytorch_lightning_runner_with_partial_kwargs(modelstore, bias_pair):

    N, D_in, H, D_out = 64, 1000, 100, 1
    x = torch.randn(N, D_in)
    model = ExtendedModel(D_in, H, D_out)

    tag = bentoml.pytorch_lightning.save(
        "pytorch_test_extended", model, model_store=modelstore
    )
    bias1, bias2 = bias_pair
    runner1 = bentoml.pytorch_lightning.load_runner(
        tag, model_store=modelstore, partial_kwargs=dict(bias=bias1)
    )

    runner2 = bentoml.pytorch_lightning.load_runner(
        tag, model_store=modelstore, partial_kwargs=dict(bias=bias2)
    )

    res1 = runner1.run_batch(x)[0][0].item()
    res2 = runner2.run_batch(x)[0][0].item()

    # tensor to float may introduce larger errors, so we bump rel_tol
    # from 1e-9 to 1e-6 just in case
    assert math.isclose(res1 - res2, bias1 - bias2, rel_tol=1e-6)
