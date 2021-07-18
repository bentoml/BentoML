import os

import coremltools as ct
import pytest
import torch
from coremltools.models import MLModel

from bentoml._internal.exceptions import InvalidArgument
from bentoml.coreml import CoreMLModel
from tests._internal.bento_services.pytorch import LinearModel, mock_df


def pytorch_to_coreml(pytorch_model: LinearModel) -> "MLModel":
    """
    CoreML is not for training ML models but rather for converting pretrained models
    and running them on Apple devices. Therefore, in this train we convert the
    pretrained LinearModel from the tests._internal.bento_services.pytorch
    module into a CoreML module.
    """
    pytorch_model.eval()
    traced_model = torch.jit.trace(pytorch_model, torch.Tensor(mock_df.values))
    return ct.convert(
        traced_model, inputs=[ct.TensorType(name='input', shape=mock_df.shape)]
    )


def test_coreml_save_load(tmpdir):
    model = pytorch_to_coreml(LinearModel())
    coreml_artifact = CoreMLModel(model)
    coreml_artifact.save(tmpdir)
    assert os.path.exists(
        CoreMLModel.get_path(tmpdir, CoreMLModel.COREMLMODEL_FILE_EXTENSION)
    )

    coreml_loaded = CoreMLModel.load(tmpdir)
    assert isinstance(coreml_loaded, MLModel)


def test_invalid_coreml_load(tmpdir):
    with pytest.raises(InvalidArgument):
        CoreMLModel.load(tmpdir)
