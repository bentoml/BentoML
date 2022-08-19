from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml._internal.configuration.containers import BentoMLConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from simple_di.providers import ConfigDictType


@pytest.fixture(scope="function", name="config_cls")
def get_bentomlconfiguration_from_str(
    tmp_path: Path,
) -> t.Callable[[str], ConfigDictType]:
    def inner(config: str) -> ConfigDictType:
        path = tmp_path / "configuration.yaml"
        path.write_text(config)
        return BentoMLConfiguration(override_config_file=path.__fspath__()).as_dict()

    return inner


@pytest.mark.usefixtures("config_cls")
def test_backward_configuration(config_cls: t.Callable[[str], ConfigDictType]):
    OLD_CONFIG = """\
api_server:
    backlog: 4096
    max_request_size: 8624612341
"""
    bentoml_cfg = config_cls(OLD_CONFIG)
    assert "backlog" not in bentoml_cfg["api_server"]
    assert "max_request_size" not in bentoml_cfg["api_server"]
    assert "cors" not in bentoml_cfg["api_server"]
    assert bentoml_cfg["api_server"]["http"]["backlog"] == 4096
    assert bentoml_cfg["api_server"]["http"]["max_request_size"] == 8624612341


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
