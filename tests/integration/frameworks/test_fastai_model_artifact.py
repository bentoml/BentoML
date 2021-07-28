import numpy as np
import pandas as pd
from fastai.data.block import DataBlock
from fastai.learner import Learner
from fastai.torch_core import Module

from bentoml.fastai import FastAIModel
from tests._internal.frameworks.pytorch_utils import LinearModel, test_df
from tests._internal.helpers import assert_have_file_extension


def get_items(_x):
    return np.ones([5, 5], np.float32)


class Loss(Module):
    reduction = "none"

    def forward(self, x, _y):
        return x

    def activation(self, x):
        return x

    def decodes(self, x):
        return x


def pack_models(path: str) -> None:

    model = LinearModel()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)

    FastAIModel(learner).save(path)


def predict_df(model: "Learner", df: "pd.DataFrame"):
    input_df = df.to_numpy().astype(np.float32)
    _, _, res = model.predict(input_df)
    return res.squeeze().item()


def test_fastai_save_pack(tmpdir):
    pack_models(tmpdir)
    assert_have_file_extension(tmpdir, ".pkl")

    loaded_fastai: "Learner" = FastAIModel.load(tmpdir)
    assert predict_df(loaded_fastai, test_df) == 5.0
