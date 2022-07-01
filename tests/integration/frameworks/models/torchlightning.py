from __future__ import annotations

import numpy as np
import torch
import pandas as pd
import torch.nn
import pytorch_lightning as pl

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.torcglightning


from .torchscript import test_y
from .torchscript import test_x_list

if torch.cuda.is_available():
    torch_x_list.append(torch.Tensor(test_x_nda, device="cuda"))


def generate_models():
    class LightningLinearModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 1, bias=False)
            torch.nn.init.ones_(self.linear.weight)

        def forward(self, x):
            return self.linear(x)

    yield LightningLinearModel()


models = [
    FrameworkTestModel(
        name="torchscript",
        model=model,
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[x],
                            expected=lambda out: out == test_y,
                        )
                        for x in test_x_list
                    ],
                },
            ),
        ],
    )
    for model in generate_models()
]
