from __future__ import annotations

import typing as t
from tempfile import NamedTemporaryFile

from bentoml._internal.configuration.containers import BentoMLConfiguration


def create_configuration_from_string(config_str: str) -> dict[str, t.Any]:
    tmpfile = NamedTemporaryFile(mode="w+", delete=False)
    tmpfile.write(config_str)
    tmpfile.flush()
    tmpfile.close()

    return BentoMLConfiguration(override_config_file=tmpfile.name).asdict()


def test_bentoml_configuration_runner_override():
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

    bentoml_cfg = create_configuration_from_string(OVERRIDE_RUNNERS)
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


def test_runner_gpu_configuration():
    GPU_INDEX = """\
runners:
    resources:
        nvidia.com/gpu: [1, 2, 4]
"""
    bentoml_cfg = create_configuration_from_string(GPU_INDEX)
    assert bentoml_cfg["runners"]["resources"] == {"nvidia.com/gpu": [1, 2, 4]}

    GPU_INDEX_WITH_STRING = """\
runners:
    resources:
        nvidia.com/gpu: "[1, 2, 4]"
"""
    bentoml_cfg = create_configuration_from_string(GPU_INDEX_WITH_STRING)
    # this behaviour can be confusing
    assert bentoml_cfg["runners"]["resources"] == {"nvidia.com/gpu": "[1, 2, 4]"}
