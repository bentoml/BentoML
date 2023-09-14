from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image

import time

import numpy as np
from utils import exponential_buckets

import bentoml

mnist_model = bentoml.pytorch.get("mnist_cnn:latest")
_BuiltinRunnable = mnist_model.to_runnable()

inference_duration = bentoml.metrics.Histogram(
    name="inference_duration",
    documentation="Duration of inference",
    labelnames=["torch_version", "device_id"],
    buckets=exponential_buckets(0.001, 1.5, 10.0),
)


class CustomMnistRunnable(_BuiltinRunnable):
    def __init__(self):
        super().__init__()
        import torch

        print("Running on device:", self.device_id)
        self.torch_version = torch.__version__
        print("Running on torch version:", self.torch_version)

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        output = super().__call__(input_data)
        inference_duration.labels(
            torch_version=self.torch_version, device_id=self.device_id
        ).observe(time.perf_counter() - start)
        return output.argmax(dim=1)


mnist_runner = bentoml.Runner(
    CustomMnistRunnable,
    method_configs={"__call__": {"max_batch_size": 50, "max_latency_ms": 600}},
)

svc = bentoml.Service(
    "pytorch_mnist_demo", runners=[mnist_runner], models=[mnist_model]
)


@svc.api(input=bentoml.io.Image(), output=bentoml.io.NumpyNdarray())
async def predict(image: PIL.Image.Image) -> np.ndarray:
    arr = np.array(image).reshape([-1, 1, 28, 28])
    res = await mnist_runner.async_run(arr)
    return res.numpy()
