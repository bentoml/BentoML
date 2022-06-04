import pandas as pd

import paddle
import paddle.nn as nn
from paddle.static import InputSpec

IN_FEATURES = 13
OUT_FEATURES = 1

test_df = pd.DataFrame(
    [
        [
            -0.0405441,
            0.06636364,
            -0.32356227,
            -0.06916996,
            -0.03435197,
            0.05563625,
            -0.03475696,
            0.02682186,
            -0.37171335,
            -0.21419304,
            -0.33569506,
            0.10143217,
            -0.21172912,
        ]
    ]
)


class LinearModel(nn.Layer):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[IN_FEATURES], dtype="float32")])
    def forward(self, x):
        return self.fc(x)
