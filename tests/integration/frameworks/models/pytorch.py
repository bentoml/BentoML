from __future__ import annotations

import typing as t

import numpy as np
import torch
import torch.nn as nn

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.pytorch

backward_compatible = True

test_np = np.array([[1] * 5]).astype(np.float32)
test_tensor = torch.from_numpy(test_np)

expected_output = 5


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x: t.Any):
        return self.linear(x)


pytorch_model = FrameworkTestModel(
    name="pytorch",
    model=LinearModel(),
    configurations=[
        Config(
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[x],
                        expected=lambda out: out == expected_output,
                    )
                    for x in [test_np, test_tensor]
                ],
            },
        ),
    ],
)


models: list[FrameworkTestModel] = [pytorch_model]
