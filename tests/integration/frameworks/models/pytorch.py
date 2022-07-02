from __future__ import annotations

import numpy as np
import torch
import torch.nn

import bentoml
from tests.utils.frameworks.pytorch_utils import LinearModel

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.pytorch

test_np = np.array([[1] * 5]).astype(np.float32)
test_tensor = torch.from_numpy(test_np)

expected_output = 5


model = LinearModel()

pytorch_model = FrameworkTestModel(
    name="pytorch",
    model=model,
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
