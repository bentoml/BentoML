from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import pytest

from bentoml.exceptions import BentoMLConfigException
from bentoml._internal.configuration.containers import BentoMLConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture
    from simple_di.providers import ConfigDictType


@pytest.fixture(scope="function", name="config_cls")
def fixture_config_cls(tmp_path: Path) -> t.Callable[[str], ConfigDictType]:
    def inner(config: str) -> ConfigDictType:
        path = tmp_path / "configuration.yaml"
        path.write_text(config)
        return BentoMLConfiguration(override_config_file=path.__fspath__()).as_dict()

    return inner


@pytest.mark.usefixtures("config_cls")
def test_backward_configuration(
    config_cls: t.Callable[[str], ConfigDictType], caplog: LogCaptureFixture
):
    OLD_CONFIG = """\
api_server:
    max_request_size: 8624612341
    port: 5000
    host: 0.0.0.0
"""
    with caplog.at_level(logging.WARNING):
        bentoml_cfg = config_cls(OLD_CONFIG)
    assert all(
        i not in bentoml_cfg["api_server"] for i in ("max_request_size", "port", "host")
    )
    assert "cors" not in bentoml_cfg["api_server"]
    assert bentoml_cfg["api_server"]["http"]["host"] == "0.0.0.0"
    assert bentoml_cfg["api_server"]["http"]["port"] == 5000


@pytest.mark.usefixtures("config_cls")
def test_validate(config_cls: t.Callable[[str], ConfigDictType]):
    INVALID_CONFIG = """\
api_server:
    host: localhost
"""
    with pytest.raises(
        BentoMLConfigException, match="Invalid configuration file was given:*"
    ):
        config_cls(INVALID_CONFIG)


@pytest.mark.usefixtures("config_cls")
def test_backward_warning(
    config_cls: t.Callable[[str], ConfigDictType], caplog: LogCaptureFixture
):
    OLD_HOST = """\
api_server:
    host: 0.0.0.0
"""
    with caplog.at_level(logging.WARNING):
        config_cls(OLD_HOST)
    assert "field 'api_server.host' is deprecated" in caplog.text
    caplog.clear()

    OLD_PORT = """\
api_server:
    port: 4096
"""
    with caplog.at_level(logging.WARNING):
        config_cls(OLD_PORT)
    assert "field 'api_server.port' is deprecated" in caplog.text
    caplog.clear()

    OLD_MAX_REQUEST_SIZE = """\
api_server:
    max_request_size: 8624612341
"""
    with caplog.at_level(logging.WARNING):
        config_cls(OLD_MAX_REQUEST_SIZE)
    assert (
        "'api_server.max_request_size' is deprecated and has become obsolete."
        in caplog.text
    )
    caplog.clear()

    OLD_CORS = """\
api_server:
    cors:
        enabled: false
"""
    with caplog.at_level(logging.WARNING):
        config_cls(OLD_CORS)
    assert "field 'api_server.cors' is deprecated" in caplog.text
    caplog.clear()


@pytest.mark.usefixtures("config_cls")
def test_bentoml_configuration_runner_override(
    config_cls: t.Callable[[str], ConfigDictType]
):
    OVERRIDE_RUNNERS = """\
runners:
    batching:
        enabled: False
        max_batch_size: 10
    resources:
        cpu: 4
    logging:
        access:
            enabled: False
    test_runner_1:
        resources: system
    test_runner_2:
        resources:
            cpu: 2
    test_runner_gpu:
        resources:
            nvidia.com/gpu: 1
    test_runner_batching:
        batching:
            enabled: True
        logging:
            access:
                enabled: True
"""

    bentoml_cfg = config_cls(OVERRIDE_RUNNERS)
    runner_cfg = bentoml_cfg["runners"]

    # test_runner_1
    test_runner_1 = runner_cfg["test_runner_1"]
    assert test_runner_1["batching"]["enabled"] is False
    assert test_runner_1["batching"]["max_batch_size"] == 10
    assert test_runner_1["logging"]["access"]["enabled"] is False
    # assert test_runner_1["resources"]["cpu"] == 4

    # test_runner_2
    test_runner_2 = runner_cfg["test_runner_2"]
    assert test_runner_2["batching"]["enabled"] is False
    assert test_runner_2["batching"]["max_batch_size"] == 10
    assert test_runner_2["logging"]["access"]["enabled"] is False
    assert test_runner_2["resources"]["cpu"] == 2

    # test_runner_gpu
    test_runner_gpu = runner_cfg["test_runner_gpu"]
    assert test_runner_gpu["batching"]["enabled"] is False
    assert test_runner_gpu["batching"]["max_batch_size"] == 10
    assert test_runner_gpu["logging"]["access"]["enabled"] is False
    assert test_runner_gpu["resources"]["cpu"] == 4  # should use global
    assert test_runner_gpu["resources"]["nvidia.com/gpu"] == 1

    # test_runner_batching
    test_runner_batching = runner_cfg["test_runner_batching"]
    assert test_runner_batching["batching"]["enabled"] is True
    assert test_runner_batching["batching"]["max_batch_size"] == 10
    assert test_runner_batching["logging"]["access"]["enabled"] is True
    assert test_runner_batching["resources"]["cpu"] == 4  # should use global


@pytest.mark.usefixtures("config_cls")
def test_runner_gpu_configuration(config_cls: t.Callable[[str], ConfigDictType]):
    GPU_INDEX = """\
runners:
    resources:
        nvidia.com/gpu: [1, 2, 4]
"""
    bentoml_cfg = config_cls(GPU_INDEX)
    assert bentoml_cfg["runners"]["resources"] == {"nvidia.com/gpu": [1, 2, 4]}

    GPU_INDEX_WITH_STRING = """\
runners:
    resources:
        nvidia.com/gpu: "[1, 2, 4]"
"""
    bentoml_cfg = config_cls(GPU_INDEX_WITH_STRING)
    # this behaviour can be confusing
    assert bentoml_cfg["runners"]["resources"] == {"nvidia.com/gpu": "[1, 2, 4]"}


@pytest.mark.usefixtures("config_cls")
def test_runner_timeouts(config_cls: t.Callable[[str], ConfigDictType]):
    RUNNER_TIMEOUTS = """\
runners:
    timeout: 50
    test_runner_1:
        timeout: 100
    test_runner_2:
        resources: system
"""
    bentoml_cfg = config_cls(RUNNER_TIMEOUTS)
    runner_cfg = bentoml_cfg["runners"]
    assert runner_cfg["timeout"] == 50
    assert runner_cfg["test_runner_1"]["timeout"] == 100
    assert runner_cfg["test_runner_2"]["timeout"] == 50
