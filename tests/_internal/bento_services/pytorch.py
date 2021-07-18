import pandas as pd
import torch
from torch import nn

mock_df = pd.DataFrame([[1, 1, 1, 1, 1]])


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)
