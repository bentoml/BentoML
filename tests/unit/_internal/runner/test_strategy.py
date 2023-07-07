from __future__ import annotations

import typing as t

import pytest

if t.TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import bentoml
from bentoml._internal.resource import get_resource
from bentoml._internal.runner import strategy
from bentoml._internal.runner.strategy import DefaultStrategy


class GPURunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)


class SingleThreadRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False


class MultiThreadRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True


def unvalidated_get_resource(x: t.Dict[str, t.Any], y: str):
    return get_resource(x, y, validate=False)


@pytest.mark.parametrize("env", ["", "-1"])
def test_gpu_strategy_with_env_disable(monkeypatch: MonkeyPatch, env: str):
    with monkeypatch.context() as mcls:
        mcls.setattr(strategy, "get_resource", unvalidated_get_resource)
        mcls.setenv("CUDA_VISIBLE_DEVICES", env)
        assert pytest.raises(
            AttributeError, DefaultStrategy.get_worker_count, GPURunnable, None, 1
        )


def test_gpu_strategy_from_env(monkeypatch: MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setattr(strategy, "get_resource", unvalidated_get_resource)
        mcls.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        assert DefaultStrategy.get_worker_count(GPURunnable, None, 1) == 3
        assert DefaultStrategy.get_worker_count(GPURunnable, None, 0.5) == 2
        assert DefaultStrategy.get_worker_count(GPURunnable, None, 0.33) == 1


def test_gpu_strategy_respect_minus_spec(monkeypatch: MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setattr(strategy, "get_resource", unvalidated_get_resource)
        mcls.setenv("CUDA_VISIBLE_DEVICES", "0,1,-1,2")
        assert (
            DefaultStrategy.get_worker_count(
                GPURunnable, {"nvidia.com/gpu": "system"}, 1
            )
            == 2
        )
        assert (
            DefaultStrategy.get_worker_count(
                GPURunnable, {"nvidia.com/gpu": "system"}, 0.5
            )
            == 1
        )


def test_gpu_strategy_worker_env(monkeypatch: MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setattr(strategy, "get_resource", unvalidated_get_resource)
        mcls.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        envs = DefaultStrategy.get_worker_env(
            GPURunnable, {"nvidia.com/gpu": "system"}, 1, 0
        )
        assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
        envs = DefaultStrategy.get_worker_env(
            GPURunnable, {"nvidia.com/gpu": "system"}, 1, 1
        )
        assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
        envs = DefaultStrategy.get_worker_env(
            GPURunnable, {"nvidia.com/gpu": "system"}, 1, 2
        )
        assert envs.get("CUDA_VISIBLE_DEVICES") == "2"


def test_gpu_strategy_worker_env_respect_minus(monkeypatch: MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setattr(strategy, "get_resource", unvalidated_get_resource)
        mcls.setenv("CUDA_VISIBLE_DEVICES", "0,-1,1,2")
        envs = DefaultStrategy.get_worker_env(
            GPURunnable, {"nvidia.com/gpu": "system"}, 1, 0
        )
        assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
        assert pytest.raises(
            ValueError,
            DefaultStrategy.get_worker_env,
            GPURunnable,
            {"nvidia.com/gpu": "system"},
            1,
            1,
        )


def test_gpu_strategy_worker_env_respect_literal(monkeypatch: MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setattr(strategy, "get_resource", unvalidated_get_resource)
        mcls.setenv("CUDA_VISIBLE_DEVICES", "GPU-5ebe9f43-ac33420d4628")
        envs = DefaultStrategy.get_worker_env(
            GPURunnable, {"nvidia.com/gpu": "system"}, 1, 0
        )
        assert envs.get("CUDA_VISIBLE_DEVICES") == "GPU-5ebe9f43-ac33420d4628"
        assert pytest.raises(
            ValueError,
            DefaultStrategy.get_worker_env,
            GPURunnable,
            {"nvidia.com/gpu": "system"},
            1,
            1,
        )


def test_default_gpu_strategy(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    assert DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 2}, 1) == 2
    assert DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 2}, 2) == 4
    assert pytest.raises(
        ValueError,
        DefaultStrategy.get_worker_count,
        GPURunnable,
        {"nvidia.com/gpu": 0},
        1,
    )
    assert (
        DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 1)
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 2)
        == 4
    )

    assert (
        DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 0.5)
        == 1
    )
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": [2, 7, 9]}, 0.5
        )
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": [2, 7, 8, 9]}, 0.5
        )
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": [2, 5, 7, 8, 9]}, 0.4
        )
        == 2
    )

    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 1, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 1, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 1, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"

    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 2, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 2, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 2, 2)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 2, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 2, 2)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"

    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7]}, 0.5, 0
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,7"

    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7, 8, 9]}, 0.5, 0
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,7"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7, 8, 9]}, 0.5, 1
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "8,9"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7, 8, 9]}, 0.25, 0
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,7,8,9"

    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 6, 7, 8, 9]}, 0.4, 0
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,6"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 6, 7, 8, 9]}, 0.4, 1
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7,8"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 6, 7, 8, 9]}, 0.4, 2
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "9"


def test_default_cpu_strategy(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    assert DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 2}, 1) == 2
    assert DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 0.5}, 1) == 1
    assert DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 2}, 2) == 4
    assert DefaultStrategy.get_worker_count(MultiThreadRunnable, {"cpu": 4}, 1) == 1
    assert DefaultStrategy.get_worker_count(MultiThreadRunnable, {"cpu": 4}, 2) == 2

    envs = DefaultStrategy.get_worker_env(SingleThreadRunnable, {"cpu": 4}, 1, 0)
    assert envs["CUDA_VISIBLE_DEVICES"] == "-1"
    assert envs["BENTOML_NUM_THREAD"] == "1"

    envs = DefaultStrategy.get_worker_env(SingleThreadRunnable, {"cpu": 4}, 1, 1)
    assert envs["BENTOML_NUM_THREAD"] == "1"

    envs = DefaultStrategy.get_worker_env(MultiThreadRunnable, {"cpu": 4}, 1, 0)
    assert envs["BENTOML_NUM_THREAD"] == "4"
    envs = DefaultStrategy.get_worker_env(MultiThreadRunnable, {"cpu": 3.5}, 1, 1)
    assert envs["BENTOML_NUM_THREAD"] == "4"
