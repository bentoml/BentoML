from __future__ import annotations

import typing as t

import pytest

if t.TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import bentoml
from bentoml._internal.runner import strategy
from bentoml._internal.resource import get_resource
from bentoml._internal.runner.strategy import DefaultStrategy


class GPURunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)


def unvalidated_get_resource(x: t.Dict[str, t.Any], y: str):
    return get_resource(x, y, validate=False)


def test_default_gpu_strategy(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    assert DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 2}) == 2
    assert pytest.raises(
        ValueError, DefaultStrategy.get_worker_count, GPURunnable, {"nvidia.com/gpu": 0}
    )
    assert (
        DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": [2, 7]}) == 2
    )

    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": 2}, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = DefaultStrategy.get_worker_env(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"
