import json
import uuid
import pytest

import torch
import pandas as pd


import bentoml
from bentoml.utils.tempdir import TempDirectory
from tests.bento_service_examples.pytorch_lightning_classifier import (
    PytorchLightningService,
)
from pytorch_lightning.core.lightning import LightningModule
from tests.integration.utils import (
    run_api_server_docker_container,
    build_api_server_docker_image,
)


class TorchLightningModel(LightningModule):
    def forward(self, x):
        return x.add(1)


class TorchLightningModelTwo(LightningModule):
    def forward(self, x):
        return x.add(2)


def test_pytorch_lightning_model_artifact_with_saved_lightning_model():
    with TempDirectory() as temp_dir:
        svc = PytorchLightningService()
        model = TorchLightningModel()
        script = model.to_torchscript()
        script_path = f'{temp_dir}/model.pt'
        torch.jit.save(script, script_path)
        svc.pack('model', script_path)

        saved_path = svc.save(version=uuid.uuid4().hex[0:8])
        svc = bentoml.load(saved_path)
        result = svc.predict(pd.DataFrame([[5, 4, 3, 2]]))
        assert result.tolist() == [[6, 5, 4, 3]]


def test_pytorch_lightning_model_artifact():
    svc = PytorchLightningService()
    model = TorchLightningModel()
    svc.pack('model', model)

    saved_path = svc.save(version=uuid.uuid4().hex[0:8])
    svc = bentoml.load(saved_path)
    result = svc.predict(pd.DataFrame([[5, 4, 3, 2]]))
    assert result.tolist() == [[6, 5, 4, 3]]


@pytest.fixture(scope='module')
def image(clean_context):
    svc = PytorchLightningService()
    model = TorchLightningModelTwo()
    svc.pack('model', model)
    saved_path = svc.save(version=uuid.uuid4().hex[0:8])

    yield clean_context.enter_context(build_api_server_docker_image(saved_path))


@pytest.fixture(scope='module')
def host(image):
    with run_api_server_docker_container(image, timeout=500) as host:
        yield host


@pytest.mark.asyncio
async def test_pytorch_lightning_with_docker(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict",
        headers=(("Content-Type", "application/json"),),
        data=json.dumps([[5, 4, 3, 2]]),
        assert_status=200,
        assert_data=b'[[7, 6, 5, 4]]',
    )
