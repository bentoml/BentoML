from __future__ import annotations

import typing as t

import torch
import helpers

import bentoml

if t.TYPE_CHECKING:
    from PIL.Image import Image

# triton runner
triton_runner = bentoml.triton.Runner(
    "triton_runner",
    "./model_repository",
    cli_args=["--model-control-mode=explicit", "--load-model=onnx_yolov5s"],
)

bentoml_yolov5_onnx = (
    bentoml.onnx.get("onnx-yolov5")
    .with_options(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    .to_runner()
)

svc = bentoml.Service(
    "triton-integration-onnx", runners=[triton_runner, bentoml_yolov5_onnx]
)


@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def bentoml_onnx_yolov5_infer(im: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        im, 640, False, "cpu" if not torch.cuda.is_available() else "cuda:0", False, 32
    )
    y = await bentoml_yolov5_onnx.async_run(prep.to_numpy())
    return helpers.postprocess_yolov5_prediction(y, prep)


@svc.api(
    input=bentoml.io.Image.from_sample("./data/zidane.jpg"), output=bentoml.io.JSON()
)
async def triton_onnx_yolov5_infer(im: Image) -> dict[str, str]:
    prep = helpers.prepare_yolov5_input(
        im, 640, False, "cpu" if not torch.cuda.is_available() else "cuda:0", False, 32
    )
    InferResult = await triton_runner.onnx_yolov5s.async_run(prep.to_numpy())
    return helpers.postprocess_yolov5_prediction(InferResult.as_numpy("output0"), prep)


# Triton Model management API
@svc.api(
    input=bentoml.io.JSON.from_sample({"model_name": "onnx_yolov5s"}),
    output=bentoml.io.JSON(),
)
async def model_config(input_model: dict[t.Literal["model_name"], str]):
    return await triton_runner.get_model_config(input_model["model_name"], as_json=True)


@svc.api(input=bentoml.io.Text.from_sample("onnx_yolov5s"), output=bentoml.io.JSON())
async def unload_model(input_model: str):
    await triton_runner.unload_model(input_model)
    return {"unloaded": input_model}


@svc.api(input=bentoml.io.Text.from_sample("onnx_yolov5s"), output=bentoml.io.JSON())
async def load_model(input_model: str):
    await triton_runner.load_model(input_model)
    return {"loaded": input_model}


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def list_models(_: str) -> list[str]:
    resp = await triton_runner.get_model_repository_index()
    return [i.name for i in resp.models]
