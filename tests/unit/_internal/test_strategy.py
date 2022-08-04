import os

import bentoml
from bentoml._internal.runner.strategy import DefaultStrategy


class GPURunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)


def test_default_gpu_strategy():
    DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 2}) == 2
    DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": 0}) == 0
    DefaultStrategy.get_worker_count(GPURunnable, {"nvidia.com/gpu": [2, 7]}) == 2

    DefaultStrategy.setup_worker(GPURunnable, {"nvidia.com/gpu": 2}, 1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    DefaultStrategy.setup_worker(GPURunnable, {"nvidia.com/gpu": 2}, 2)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    DefaultStrategy.setup_worker(GPURunnable, {"nvidia.com/gpu": [2, 7]}, 2) == 7
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7"
