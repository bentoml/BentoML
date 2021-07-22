import os

import pandas as pd
import pytorch_lightning as pl
import torch

from bentoml.pytorch import PyTorchLightningModel

test_df = pd.DataFrame([[5, 4, 3, 2]])


class FooModel(pl.LightningModule):
    # fmt: off
    def forward(self, input): return input.add(1)
    # fmt: on


def predict_df(model: pl.LightningModule, df: pd.DataFrame):
    input_tensor = torch.from_numpy(df.to_numpy())
    return model(input_tensor).numpy().tolist()


def test_pl_save_load(tmpdir):
    model: pl.LightningModule = FooModel()
    PyTorchLightningModel(model).save(tmpdir)
    assert os.path.exists(PyTorchLightningModel.get_path(tmpdir, '.pt'))

    pl_loaded: pl.LightningModule = PyTorchLightningModel.load(tmpdir)

    assert (
        predict_df(model, test_df) == predict_df(pl_loaded, test_df) == [[6, 5, 4, 3]]
    )
