from __future__ import annotations

import numpy as np
import torch
import pandas as pd
import torch.nn

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.torchscript

backward_compatible = True

test_x_nda = np.array([[1] * 5])
test_x_df = pd.DataFrame(test_x_nda)
test_x = torch.Tensor(test_x_nda, device="cpu")

test_x_list = [test_x_nda, test_x_df, test_x]
test_y = 5


if torch.cuda.is_available():
    test_x_list.append(torch.Tensor(test_x_nda, device="cuda"))


def generate_models():
    class NNLinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 1, bias=False)
            torch.nn.init.ones_(self.linear.weight)

        def forward(self, x):
            return self.linear(x)

    nn_model = NNLinearModel()

    tracing_inp = torch.ones(5)
    yield torch.jit.trace(nn_model, tracing_inp)

    yield torch.jit.script(nn_model)


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
                        ),
                    ],
                },
            ),
        ],
    )
    for model in generate_models()
    for x in test_x_list
]
