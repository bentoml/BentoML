from __future__ import annotations

import typing as t

import numpy as np
import torch

import bentoml
from bentoml._internal.types import LazyType
from bentoml._internal.utils import LazyLoader

if t.TYPE_CHECKING:
    import numpy as np
    import helpers
    from PIL.Image import Image
    from numpy.typing import NDArray
else:
    np = LazyLoader("np", globals(), "numpy")
    helpers = LazyLoader("helpers", globals(), "helpers")

# triton runner
triton_runner = bentoml.triton.Runner("triton_runner", "./model_repository")

# yolov5
yolov5_torchscript_model = bentoml.torchscript.get("torchscript-yolov5")

# shared metadata, stride and names
metadata: dict[str, t.Any] = yolov5_torchscript_model.info.metadata["model_info"]
stride: int = metadata["stride"]
names: dict[int, str] = metadata["names"]

bentoml_yolov5_torchscript = yolov5_torchscript_model.to_runner()
bentoml_yolov5_tensorflow = bentoml.tensorflow.get("tensorflow-yolov5").to_runner()
bentoml_yolov5_onnx = (
    bentoml.onnx.get("onnx-yolov5")
    .with_options(
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    .to_runner()
)

bentoml_mnist_torchscript = bentoml.torchscript.get("torchscript-mnist").to_runner()
bentoml_mnist_tensorflow = bentoml.tensorflow.get("tensorflow-mnist").to_runner()
bentoml_mnist_onnx = (
    bentoml.onnx.get("onnx-mnist")
    .with_options(
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    .to_runner()
)

svc = bentoml.Service(
    "triton-integration",
    runners=[
        triton_runner,
        bentoml_mnist_onnx,
        bentoml_mnist_torchscript,
        bentoml_mnist_tensorflow,
        bentoml_yolov5_onnx,
        bentoml_yolov5_torchscript,
        bentoml_yolov5_tensorflow,
    ],
)


# Comparison between Triton and bentoml.Runner
preprocess_device = "cpu"

#### BentoML YOLOv5
@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def bentoml_torchscript_yolov5_infer(fp: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        fp, 640, False, preprocess_device, False, stride
    )
    y = await bentoml_yolov5_torchscript.async_run(prep.im)
    return helpers.postprocess_yolov5_prediction(y, prep, names)


@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def bentoml_tensorflow_yolov5_infer(fp: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        fp, 640, False, preprocess_device, False, stride
    )
    im = prep.im
    _, _, h, w = prep.im.shape
    im = prep.im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192, 3)
    y = await bentoml_yolov5_tensorflow.async_run(im.cpu().numpy())
    y = [
        _
        if LazyType["NDArray[np.float32]"]("numpy", "ndarray").isinstance(_)
        else _.numpy()
        for _ in y
    ]
    y[0][..., :4] *= [w, h, w, h]
    return helpers.postprocess_yolov5_prediction(y, prep, names)


@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def bentoml_onnx_yolov5_infer(im: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        im, 640, False, preprocess_device, False, stride
    )
    y = await bentoml_yolov5_onnx.async_run(prep.to_numpy())

    return helpers.postprocess_yolov5_prediction(y, prep, names)


### Triton YOLOv5
@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"),
    output=bentoml.io.JSON(),
)
async def triton_torchscript_yolov5_infer(im: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        im, 640, False, preprocess_device, False, stride
    )
    InferResult = await triton_runner.torchscript_yolov5s.async_run(prep.to_numpy())
    return helpers.postprocess_yolov5_prediction(
        InferResult.as_numpy("OUTPUT__0"), prep, names
    )


@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def triton_tensorflow_yolov5_infer(fp: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        fp, 640, False, preprocess_device, False, stride
    )
    im = prep.im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192, 3)
    InferResult = await triton_runner.tensorflow_yolov5s.async_run(im.cpu().numpy())
    return helpers.postprocess_yolov5_prediction(
        InferResult.as_numpy("output_0"), prep, names
    )


@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def triton_onnx_yolov5_infer(im: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        im, 640, False, preprocess_device, False, stride
    )
    InferResult = await triton_runner.onnx_yolov5s.async_run(prep.to_numpy())
    return helpers.postprocess_yolov5_prediction(
        InferResult.as_numpy("output0"), prep, names
    )


#### BentoML MNIST
@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def bentoml_torchscript_mnist_infer(im: Image) -> NDArray[t.Any]:
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    res = await bentoml_mnist_torchscript.async_run(arr)
    return res.detach().cpu().numpy()


@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def bentoml_tensorflow_mnist_infer(im: Image) -> NDArray[t.Any]:
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, (0, 3)).astype("float32")
    return await bentoml_mnist_tensorflow.async_run(arr)


@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def bentoml_onnx_mnist_infer(im: Image) -> NDArray[t.Any]:
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    return await bentoml_mnist_onnx.async_run(arr)


#### Triton MNIST


@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def triton_torchscript_mnist_infer(im: Image) -> NDArray[t.Any]:
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    InferResult = await triton_runner.torchscript_mnist.async_run(arr)
    return InferResult.as_numpy("OUTPUT__0")


@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def triton_tensorflow_mnist_infer(im: Image) -> NDArray[t.Any]:
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, (0, 3)).astype("float32")
    InferResult = await triton_runner.tensorflow_mnist.async_run(arr)
    return InferResult.as_numpy("output_0")


@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def triton_onnx_mnist_infer(im: Image) -> NDArray[t.Any]:
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    InferResult = await triton_runner.onnx_mnist.async_run(arr)
    return InferResult.as_numpy("output_0")


# Triton Model management API
@svc.api(
    input=bentoml.io.JSON.from_sample({"model_name": "onnx_mnist", "protocol": "grpc"}),
    output=bentoml.io.JSON(),
)
async def model_config(input_model: dict[t.Literal["model_name", "protocol"], str]):
    attrs: dict[str, t.Any] = {}
    protocol = input_model["protocol"]
    if protocol == "grpc":
        attrs["as_json"] = True
    return await getattr(triton_runner, f"{protocol}_get_model_config")(
        input_model["model_name"], **attrs
    )


@svc.api(input=bentoml.io.Text.from_sample("onnx_mnist"), output=bentoml.io.JSON())
async def unload_model(input_model: str):
    await triton_runner.unload_model(input_model)
    return {"unloaded": input_model}


@svc.api(input=bentoml.io.Text.from_sample("onnx_mnist"), output=bentoml.io.JSON())
async def load_model(input_model: str):
    await triton_runner.load_model(input_model)
    return {"loaded": input_model}


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def list_models(_: str) -> list[str]:
    resp = await triton_runner.get_model_repository_index()
    return [i.name for i in resp.models]
