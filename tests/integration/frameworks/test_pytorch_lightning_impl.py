import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

import bentoml.pytorch_lightning
from tests.utils.helpers import assert_have_file_extension

test_df = pd.DataFrame([[5, 4, 3, 2]])


class AdditionModel(pl.LightningModule):
    def forward(self, inputs):
        return inputs.add(1)


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
    runner._setup()
    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert torch.get_num_threads() == runner.num_concurrency_per_replica

    res = runner._run_batch(torch.from_numpy(test_df.to_numpy()))
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
    runner._setup()
    assert runner.num_concurrency_per_replica == 1
    assert torch.cuda.device_count() == runner.num_replica
