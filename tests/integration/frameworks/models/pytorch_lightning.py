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

framework = bentoml.pytorch_lightning

backward_compatible = True

test_x_nda = np.array([[1] * 5])
test_x_df = pd.DataFrame(test_x_nda)
test_x = torch.Tensor(test_x_nda, device="cpu")

test_x_list = [test_x_nda, test_x_df, test_x]


def generate_models():
    class LightningLinearModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 1, bias=False)
            torch.nn.init.ones_(self.linear.weight)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    yield LightningLinearModel()


models = [
    FrameworkTestModel(
        name="pytorch_lightning",
        model=model,
        configurations=[
            Config(
                test_inputs={
                    "__call__": [
                        Input(
                            input_args=[x],
                            expected=lambda out: out == 5,
                        )
                        for x in test_x_list
                    ],
                },
            ),
        ],
    )
    for model in generate_models()
]
