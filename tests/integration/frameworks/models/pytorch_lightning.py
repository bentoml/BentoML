from __future__ import annotations

import torch
import torch.nn
import pytorch_lightning as pl

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config
from .torchscript import test_y
from .torchscript import test_x_list

framework = bentoml.pytorch_lightning

backward_compatible = True


def generate_models():
    class LightningLinearModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 1, bias=False)
            torch.nn.init.ones_(self.linear.weight)

        def forward(self, x: torch.Tensor):  # noqa
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
                            expected=lambda out: out == test_y,
                        )
                    ],
                },
            ),
        ],
    )
    for model in generate_models()
    for x in test_x_list
]
