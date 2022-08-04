import os
import typing as t

import bentoml
from bentoml._internal.runner import strategy
from bentoml._internal.resource import get_resource as resource_get_resource
from bentoml._internal.runner.strategy import DefaultStrategy


class GPURunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)


def unvalidated_get_resource(x: t.Dict[str, t.Any], y: str):
    return resource_get_resource(x, y, validate=False)


def test_default_gpu_strategy(monkeypatch):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 2}) == 2
    DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 0}) == 0
    DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": [2, 7]}) == 2

    DefaultStrategy.setup_worker(GPURunnable, {"nvidia.com/gpu": 2}, 1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    DefaultStrategy.setup_worker(GPURunnable, {"nvidia.com/gpu": 2}, 2)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    DefaultStrategy.setup_worker(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 2)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7"
