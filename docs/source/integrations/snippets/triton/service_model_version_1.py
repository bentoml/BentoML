from __future__ import annotations

import typing as t

import bentoml
import numpy as np

from tritonclient.grpc.aio import InferInput, InferRequestedOutput, np_to_triton_dtype

if t.TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL.Image import Image

# triton runner
triton_runner = bentoml.triton.Runner("triton_runner", "./model_repository")

svc = bentoml.Service("triton-integration", runners=[triton_runner])


@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def predict_v1(input_data: Image) -> NDArray[t.Any]:
    arr = np.array(input_data) / 255.0
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    input_0 = InferInput("input_0", arr.shape, np_to_triton_dtype(arr.dtype))
    input_0.set_data_from_numpy(arr)
    output_0 = InferRequestedOutput("output_0")
    InferResult = await triton_runner.infer(
        "onnx_mnist", inputs=[input_0], model_version="1", outputs=[output_0]
    )
    return InferResult.as_numpy("output_0")
