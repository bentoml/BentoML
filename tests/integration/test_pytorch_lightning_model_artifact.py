import pytest
import pandas as pd


import bentoml
from tests.bento_service_examples.pytorch_lightning_classifier import (
    PytorchLightningService,
)
from tests.integration.api_server.conftest import export_service_bundle
from pytorch_lightning.core.lightning import LightningModule


class TorchLightningModel(LightningModule):
    def forward(self, x):
        return x.add(1)


@pytest.fixture(scope='module')
def lightning_svc():
    svc = PytorchLightningService()
    model = TorchLightningModel()
    svc.pack('model', model)
    return svc


def test_pytorch_lightning_model_artifact(lightning_svc):
    with export_service_bundle(lightning_svc) as saved_path:
        svc = bentoml.load(saved_path)
        result = svc.predict(pd.DataFrame([[5, 4, 3, 2]]))
        assert result.tolist() == [[6, 5, 4, 3]]
