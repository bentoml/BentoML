import torch
import pandas as pd


import bentoml
from bentoml.utils.tempdir import TempDirectory
from tests.bento_service_examples.pytorch_lightning_classifier import (
    PytorchLightningService,
)
from pytorch_lightning.core.lightning import LightningModule
from tests.integration.utils import export_service_bundle


class TorchLightningModel(LightningModule):
    def forward(self, x):
        return x.add(1)


def test_pytorch_lightning_model_artifact_with_saved_lightning_model():
    with TempDirectory() as temp_dir:
        svc = PytorchLightningService()
        model = TorchLightningModel()
        script = model.to_torchscript()
        script_path = f'{temp_dir}/model.pt'
        torch.jit.save(script, script_path)
        svc.pack('model', script_path)

        saved_path = svc.save()
        svc = bentoml.load(saved_path)
        result = svc.predict(pd.DataFrame([[5, 4, 3, 2]]))
        assert result.tolist() == [[6, 5, 4, 3]]


def test_pytorch_lightning_model_artifact():
    svc = PytorchLightningService()
    model = TorchLightningModel()
    svc.pack('model', model)

    with export_service_bundle(svc) as saved_path:
        svc = bentoml.load(saved_path)
        result = svc.predict(pd.DataFrame([[5, 4, 3, 2]]))
        assert result.tolist() == [[6, 5, 4, 3]]
