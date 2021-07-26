import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from bentoml.pytorch import PyTorchModel
from tests._internal.frameworks.pytorch_utils import LinearModel, test_df
from tests._internal.helpers import assert_have_file_extension


def predict_df(model: nn.Module, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float32)
    input_tensor = torch.from_numpy(input_data)
    return model(input_tensor).unsqueeze(dim=0).item()


@pytest.mark.parametrize("test_type", ["", "tracedmodel", "scriptedmodel"])
def test_pytorch_save_load(test_type, tmpdir):
    _model: nn.Module = LinearModel()
    if "trace" in test_type:
        tracing_inp = torch.ones(5)
        model = torch.jit.trace(_model, tracing_inp)
    elif "script" in test_type:
        model = torch.jit.script(_model)
    else:
        model = _model
    PyTorchModel(model).save(tmpdir)
    assert_have_file_extension(tmpdir, ".pt")

    pytorch_loaded: nn.Module = PyTorchModel.load(tmpdir)
    assert predict_df(model, test_df) == predict_df(pytorch_loaded, test_df)
