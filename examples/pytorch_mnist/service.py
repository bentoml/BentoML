import typing as t

import numpy as np
import PIL.Image
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray

mnist_runner = bentoml.pytorch.get("pytorch_mnist").to_runner()

svc = bentoml.Service(
    name="pytorch_mnist_demo",
    runners=[
        mnist_runner,
    ],
)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


@svc.api(
    input=NumpyNdarray(dtype="float32", enforce_dtype=True),
    output=NumpyNdarray(dtype="int64"),
)
async def predict_ndarray(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.shape == (28, 28)
    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension. Then we will also add one batch
    # dimension
    inp = np.expand_dims(inp, (0, 1))
    output_tensor = await mnist_runner.async_run(inp)
    return to_numpy(output_tensor)


@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    arr = np.array(f)/255.0
    assert arr.shape == (28, 28)

    # We are using greyscale image and our PyTorch model expect one
    # extra channel dimension. Then we will also add one batch
    # dimension
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    output_tensor = await mnist_runner.async_run(arr)
    return to_numpy(output_tensor)
