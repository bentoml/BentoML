import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai.torch_core import Module

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
