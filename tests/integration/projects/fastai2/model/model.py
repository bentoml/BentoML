import pathlib
import sys

import numpy as np
import torch
from fastai.data.block import DataBlock
from fastai.learner import Learner
from fastai.torch_core import Module
from torch import nn


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
    reduction = 'none'

    def forward(self, x, _y):
        return x

    def activation(self, x):
        return x

    def decodes(self, x):
        return x


def pack_models(path):
    from bentoml.frameworks.fastai import FastaiModelArtifact

    model = Model()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)

    FastaiModelArtifact("model").pack(learner).save(path)


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    pathlib.Path(artifacts_path).mkdir(parents=True, exist_ok=True)
    pack_models(artifacts_path)
