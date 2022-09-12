from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml.exceptions import BentoMLConfigException

if TYPE_CHECKING:
    from simple_di.providers import ConfigDictType


@pytest.mark.usefixtures("container_from_file")
def test_validate_configuration(container_from_file: t.Callable[[str], ConfigDictType]):
    CONFIG = """\
version: 1
api_server:
  http:
    host: 0.0.0.0
"""
    config = container_from_file(CONFIG)
    assert config["api_server"]["http"]["host"] == "0.0.0.0"

    INVALID_CONFIG = """\
version: 1
api_server:
  cors:
    max_age: 12345
"""
    with pytest.raises(
        BentoMLConfigException, match="Invalid configuration file was given:*"
    ):
        container_from_file(INVALID_CONFIG)


@pytest.mark.usefixtures("container_from_envvar")
def test_containers_from_envvar(
    container_from_envvar: t.Callable[[str], ConfigDictType]
):
    envvar = 'api_server.http.host="127.0.0.1" api_server.http.port=5000'
    config = container_from_envvar(envvar)
    assert config["api_server"]["http"]["host"] == "127.0.0.1"
    assert config["api_server"]["http"]["port"] == 5000


@pytest.mark.parametrize("version", [None, 1])
@pytest.mark.usefixtures("container_from_file")
def test_bentoml_configuration_runner_override(
    container_from_file: t.Callable[[str], ConfigDictType], version: int | None
):
    OVERRIDE_RUNNERS = f"""\
{'version: %d' % version if version else ''}
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

    bentoml_cfg = container_from_file(OVERRIDE_RUNNERS)
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


@pytest.mark.usefixtures("container_from_file")
def test_runner_gpu_configuration(
    container_from_file: t.Callable[[str], ConfigDictType]
):
    GPU_INDEX = """\
runners:
    resources:
        nvidia.com/gpu: [1, 2, 4]
"""
    bentoml_cfg = container_from_file(GPU_INDEX)
    assert bentoml_cfg["runners"]["resources"] == {"nvidia.com/gpu": [1, 2, 4]}

    GPU_INDEX_WITH_STRING = """\
runners:
    resources:
        nvidia.com/gpu: "[1, 2, 4]"
"""
    bentoml_cfg = container_from_file(GPU_INDEX_WITH_STRING)
    # this behaviour can be confusing
    assert bentoml_cfg["runners"]["resources"] == {"nvidia.com/gpu": "[1, 2, 4]"}


@pytest.mark.usefixtures("container_from_file")
def test_runner_timeouts(container_from_file: t.Callable[[str], ConfigDictType]):
    RUNNER_TIMEOUTS = """\
runners:
    timeout: 50
    test_runner_1:
        timeout: 100
    test_runner_2:
        resources: system
"""
    bentoml_cfg = container_from_file(RUNNER_TIMEOUTS)
    runner_cfg = bentoml_cfg["runners"]
    assert runner_cfg["timeout"] == 50
    assert runner_cfg["test_runner_1"]["timeout"] == 100
    assert runner_cfg["test_runner_2"]["timeout"] == 50
