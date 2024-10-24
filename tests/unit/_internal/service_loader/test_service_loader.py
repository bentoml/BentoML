import sys

import pytest

from bentoml._internal.bento import Bento
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.service.loader import load


@pytest.mark.usefixtures("change_test_dir", "model_store")
def test_load_service_from_directory():
    sys.modules.pop("service", None)
    service = load("./bento_with_models")
    assert service.name == "test-bento-service-with-models"
    assert len(service.models) == 2
    assert service.models[1].tag.name == "anothermodel"


@pytest.mark.usefixtures("change_test_dir", "model_store")
def test_load_service_from_bento():
    sys.modules.pop("service", None)
    build_config = BentoBuildConfig.from_file("./bento_with_models/bentofile.yaml")
    bento = Bento.create(
        build_config=build_config, version="1.0", build_ctx="./bento_with_models"
    ).save()

    sys.modules.pop("service", None)
    service = load(bento.tag)
    assert service.name == "test-bento-service-with-models"
    assert len(service.models) == 2
    assert service.models[1].tag.name == "anothermodel"
