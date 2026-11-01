from __future__ import annotations

from typing import Any

import bentoml
from bentoml._internal import configuration
from bentoml._internal.configuration.containers import BentoMLContainer


def test_inject_config_does_not_mutate_existing_config_dict(monkeypatch: Any) -> None:
    @bentoml.service(name="service_inject_config_test", traffic={"timeout": 15})
    class DummyService:
        pass

    container_dict_after_load: dict[str, Any] = {}
    original_load_config = configuration.load_config

    def mock_load_config(*args: Any, **kwargs: Any) -> None:
        original_load_config(*args, **kwargs)
        nonlocal container_dict_after_load
        container_dict_after_load = BentoMLContainer.config.get()

    monkeypatch.setattr(configuration, "load_config", mock_load_config)

    DummyService.inject_config()

    assert "api_server" not in container_dict_after_load
