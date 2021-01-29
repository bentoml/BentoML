import pytest

import torch
from torch import nn
from fastai.torch_core import Module
from fastai.learner import Learner
from fastai.data.block import DataBlock
import numpy as np
import pandas

import bentoml
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.fastai2_classifier import FastaiClassifier
from tests.integration.fastai_utils import get_items


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1, bias=False)
        torch.nn.init.ones_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


class Loss(Module):
    reduction = 'none'

    def forward(self, x, y):
        return x

    def activation(self, x):
        return x

    def decodes(self, x):
        return x


@pytest.fixture
def fastai_learner():
    model = Model()
    loss = Loss()

    dblock = DataBlock(get_items=get_items, get_y=np.sum)
    dls = dblock.datasets(None).dataloaders()
    learner = Learner(dls, model, loss)
    return learner


test_df = pandas.DataFrame([[1] * 5])


def test_fastai2_artifact_pack(fastai_learner):
    svc = FastaiClassifier()
    svc.pack('model', fastai_learner)
    assert svc.predict(test_df) == 5.0, 'Run inference before saving'

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_df) == 5.0, 'Run inference from saved model'

    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
