import numpy as np
import torch
import pandas as pd
import torch.nn as nn

test_df = pd.DataFrame([[1] * 5])


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def make_pytorch_lightning_linear_model_class():
    import pytorch_lightning as pl

    class LightningLinearModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 1, bias=False)
            torch.nn.init.ones_(self.linear.weight)

        def forward(self, x):
            return self.linear(x)

    return LightningLinearModel


def predict_df(model: nn.Module, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    return model(input_tensor).unsqueeze(dim=0).item()


class LinearModelWithBatchAxis(nn.Module):
    def __init__(self):
        super(LinearModelWithBatchAxis, self).__init__()
        self.linear = nn.Linear(5, 1, bias=False)
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, x, batch_axis=0):
        if batch_axis == 1:
            x = x.permute([1, 0])
        res = self.linear(x)
        if batch_axis == 1:
            res = res.permute([0, 1])

        return res


class ExtendedModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ExtendedModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x, bias=0.0):
        """
        In the forward function we accept a Tensor of input data and an optional bias
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred + bias
