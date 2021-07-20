import os

import coremltools as ct
import pytest
import torch

from bentoml._internal.exceptions import InvalidArgument
from bentoml.coreml import CoreMLModel
from tests.integration.frameworks.test_pytorch_model_artifact import (
    LinearModel,
    mock_df,
)


def pytorch_to_coreml(pytorch_model: LinearModel) -> "ct.models.MLModel":
    """
    CoreML is not for training ML models but rather for converting pretrained models
    and running them on Apple devices. Therefore, in this train we convert the
    pretrained LinearModel from the tests._internal.bento_services.pytorch
    module into a CoreML module.
    """
    pytorch_model.eval()
    traced_model = torch.jit.trace(pytorch_model, torch.Tensor(mock_df.values))
    model: ct.models.MLModel = ct.convert(
        traced_model, inputs=[ct.TensorType(name='input', shape=mock_df.shape)]
    )
    return model


def test_coreml_save_load(tmpdir):
    pytorch_model = LinearModel()
    model = pytorch_to_coreml(pytorch_model)
    CoreMLModel(model).save(tmpdir)
    assert os.path.exists(
        CoreMLModel.model_path(tmpdir, CoreMLModel.COREMLMODEL_FILE_EXTENSION)
    )

    coreml_loaded = CoreMLModel.load(tmpdir)
    assert isinstance(coreml_loaded, ct.models.MLModel)

    assert repr(coreml_loaded) == repr(model)


def test_invalid_coreml_load(tmpdir):
    with pytest.raises(InvalidArgument):
        CoreMLModel.load(tmpdir)
