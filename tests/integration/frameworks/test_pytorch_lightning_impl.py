import pandas as pd
import pytorch_lightning as pl
import torch

import bentoml.pytorch_lightning
from tests._internal.helpers import assert_have_file_extension

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
