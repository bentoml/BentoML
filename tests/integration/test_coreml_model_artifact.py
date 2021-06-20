import pytest

import torch
import coremltools as ct
from coremltools.models import MLModel

import bentoml
from bentoml.yatai.client import YataiClient
from tests.bento_service_examples.coreml_classifier import CoreMLClassifier
from tests.integration.test_pytorch_model_artifact import PytorchModel
from tests.integration.test_pytorch_model_artifact import test_df


@pytest.fixture()
def coreml_classifier_class():
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    CoreMLClassifier._bento_service_bundle_path = None
    CoreMLClassifier._bento_service_bundle_version = None
    return CoreMLClassifier


def convert_pytorch_to_coreml(pytorch_model: PytorchModel) -> ct.models.MLModel:
    """CoreML is not for training ML models but rather for converting pretrained models
    and running them on Apple devices. Therefore, in this train we convert the
    pretrained PytorchModel from the tests.integration.test_pytorch_model_artifact
    module into a CoreML module."""
    pytorch_model.eval()
    traced_pytorch_model = torch.jit.trace(pytorch_model, torch.Tensor(test_df.values))
    model: MLModel = ct.convert(
        traced_pytorch_model, inputs=[ct.TensorType(name="input", shape=test_df.shape)]
    )
    return model


def test_pytorch_artifact_pack(coreml_classifier_class):
    svc = coreml_classifier_class()
    pytorch_model = PytorchModel()
    model = convert_pytorch_to_coreml(pytorch_model)
    svc.pack("model", model)
    assert svc.predict(test_df) == 5.0, "Run inference before save the artifact"

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)
    assert loaded_svc.predict(test_df) == 5.0, "Run inference from saved artifact"

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f"{svc.name}:{svc.version}")
