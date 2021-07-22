import os

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from fastai.data.block import DataBlock
from fastai.learner import Learner
from fastai.torch_core import Module

from bentoml.fastai import FastaiModel

test_df = pd.DataFrame([[1] * 5])


def get_items(_x):
    return np.ones([5, 5], np.float32)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1, bias=False)
        torch.nn.init.ones_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class Loss(Module):
    reduction = "none"

    def forward(self, x, _y):
        return x

    def activation(self, x):
        return x

    def decodes(self, x):
        return x


def pack_models(path: str) -> None:

    model = Model()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)

    FastaiModel(learner).save(path)


@pytest.fixture(scope="session")
def predict_df(model: "Learner", df: "pd.DataFrame"):
    input_df = df.to_numpy().astype(np.float32)
    _, _, res = model.predict(input_df)
    return res.squeeze().item()


def test_fastai_save_pack(tmpdir, predict_df):
    pack_models(tmpdir)
    assert os.path.exists(FastaiModel.get_path(tmpdir, ".pkl"))

    loaded_fastai: "Learner" = FastaiModel.load(tmpdir)
    assert predict_df(loaded_fastai, test_df) == 5.0
