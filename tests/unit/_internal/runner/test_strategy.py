from __future__ import annotations

import typing as t

import pytest

if t.TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import bentoml
from bentoml._internal.runner import strategy
from bentoml._internal.resource import get_resource
from bentoml._internal.runner.strategy import THREAD_ENVS
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


def test_default_gpu_strategy(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    assert DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 1}, {}) == 1
    assert DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 2}, {}) == 2
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 1}
        )
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 0.5}
        )
        == 4
    )
    assert pytest.raises(
        ValueError,
        DefaultStrategy.get_worker_count,
        GPURunnable,
        {"nvidia.com/gpu": 0},
        1,
    )
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": [2, 7]}, {"nvidia.com/gpu": 1}
        )
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(
            GPURunnable, {"nvidia.com/gpu": [2, 7]}, {"nvidia.com/gpu": 0.5}
        )
        == 4
    )

    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 1}, 0
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 1}, 1
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7]}, {"nvidia.com/gpu": 1}, 1
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"

    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 0.5}, 0
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 0.5}, 1
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": 2}, {"nvidia.com/gpu": 0.5}, 2
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7]}, {"nvidia.com/gpu": 0.5}, 1
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2"
    envs = DefaultStrategy.get_worker_env(
        GPURunnable, {"nvidia.com/gpu": [2, 7]}, {"nvidia.com/gpu": 0.5}, 2
    )
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"


def test_default_cpu_strategy(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    assert DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 2}, {}) == 2
    assert (
        DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 2}, {"cpu": 1})
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 0.5}, {"cpu": 1})
        == 1
    )
    assert (
        DefaultStrategy.get_worker_count(SingleThreadRunnable, {"cpu": 2}, {"cpu": 0.5})
        == 4
    )
    assert DefaultStrategy.get_worker_count(MultiThreadRunnable, {"cpu": 4}, {}) == 1
    assert (
        DefaultStrategy.get_worker_count(MultiThreadRunnable, {"cpu": 4}, {"cpu": 4})
        == 1
    )
    assert (
        DefaultStrategy.get_worker_count(MultiThreadRunnable, {"cpu": 4}, {"cpu": 2})
        == 2
    )
    assert (
        DefaultStrategy.get_worker_count(MultiThreadRunnable, {"cpu": 4}, {"cpu": 0.5})
        == 8
    )

    envs = DefaultStrategy.get_worker_env(
        SingleThreadRunnable, {"cpu": 4}, {"cpu": 1}, 0
    )
    assert envs["CUDA_VISIBLE_DEVICES"] == "-1"
    assert all(envs[key] == "1" for key in THREAD_ENVS)
    envs = DefaultStrategy.get_worker_env(
        SingleThreadRunnable, {"cpu": 4}, {"cpu": 1}, 1
    )
    assert all(envs[key] == "1" for key in THREAD_ENVS)
    envs = DefaultStrategy.get_worker_env(
        SingleThreadRunnable, {"cpu": 4}, {"cpu": 2}, 0
    )
    assert all(envs[key] == "1" for key in THREAD_ENVS)
    envs = DefaultStrategy.get_worker_env(
        MultiThreadRunnable, {"cpu": 4}, {"cpu": 4}, 0
    )
    assert all(envs[key] == "4" for key in THREAD_ENVS)
    envs = DefaultStrategy.get_worker_env(MultiThreadRunnable, {"cpu": 3.5}, {}, 0)
    assert all(envs[key] == "4" for key in THREAD_ENVS)
    envs = DefaultStrategy.get_worker_env(
        MultiThreadRunnable, {"cpu": 3.5}, {"cpu": 2}, 0
    )
    assert all(envs[key] == "2" for key in THREAD_ENVS)
    envs = DefaultStrategy.get_worker_env(
        MultiThreadRunnable, {"cpu": 3.5}, {"cpu": 2}, 1
    )
    assert all(envs[key] == "2" for key in THREAD_ENVS)
