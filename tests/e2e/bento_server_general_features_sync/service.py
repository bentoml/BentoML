import typing as t

import numpy as np
import pandas as pd
import pydantic
from PIL.Image import Image as PILImage
from PIL.Image import fromarray

import bentoml
import bentoml.picklable_model
from bentoml.io import File
from bentoml.io import JSON
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray
from bentoml.io import PandasSeries
from bentoml.io import PandasDataFrame
from bentoml._internal.types import FileLike
from bentoml._internal.types import JSONSerializable


class _Schema(pydantic.BaseModel):
    name: str = "test"
    endpoints: t.List[str] = ["predict", "health"]


json_echo_runner = bentoml.picklable_model.load_runner(
    "sk_model",
    method_name="echo_json",
    name="json_echo_runner",
    batch=True,
)
ndarray_pred_runner = bentoml.picklable_model.load_runner(
    "sk_model",
    method_name="predict_ndarray",
    name="ndarray_pred_runner",
    batch=True,
)
dataframe_pred_runner = bentoml.picklable_model.load_runner(
    "sk_model",
    method_name="predict_dataframe",
    name="dataframe_pred_runner",
    batch=True,
)
file_pred_runner = bentoml.picklable_model.load_runner(
    "sk_model",
    method_name="predict_file",
    name="file_pred_runner",
    batch=True,
)

multi_ndarray_pred_runner = bentoml.picklable_model.load_runner(
    "sk_model",
    method_name="predict_multi_ndarray",
    name="multi_ndarray_pred_runner",
    batch=True,
)
echo_multi_ndarray_pred_runner = bentoml.picklable_model.load_runner(
    "sk_model",
    method_name="echo_multi_ndarray",
    name="echo_multi_ndarray_pred_runner",
    batch=True,
)


svc = bentoml.Service(
    name="general_sync",
    runners=[
        json_echo_runner,
        ndarray_pred_runner,
        dataframe_pred_runner,
        file_pred_runner,
        multi_ndarray_pred_runner,
        echo_multi_ndarray_pred_runner,
    ],
)


@svc.api(input=JSON(), output=JSON())
def echo_json(json_obj: JSONSerializable) -> JSONSerializable:
    return json_echo_runner.run(json_obj)


@svc.api(
    input=JSON(pydantic_model=_Schema),
    output=JSON(),
)
def pydantic_json(json_obj: JSONSerializable) -> JSONSerializable:
    return json_echo_runner.run(json_obj)


@svc.api(
    input=NumpyNdarray(shape=(2, 2), enforce_shape=True),
    output=NumpyNdarray(shape=(1, 4)),
)
def predict_ndarray_enforce_shape(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.shape == (2, 2)
    return ndarray_pred_runner.run(inp)


@svc.api(
    input=NumpyNdarray(dtype="uint8", enforce_dtype=True),
    output=NumpyNdarray(dtype="str"),
)
def predict_ndarray_enforce_dtype(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.dtype == np.dtype("uint8")
    return ndarray_pred_runner.run(inp)


@svc.api(
    input=PandasDataFrame(dtype={"col1": "int64"}, orient="records"),
    output=PandasSeries(),
)
def predict_dataframe(df: "pd.DataFrame") -> "pd.Series":
    assert df["col1"].dtype == "int64"
    output = dataframe_pred_runner.run(df)
    assert isinstance(output, pd.Series)
    return output


@svc.api(input=File(), output=File())
def predict_file(f: FileLike) -> bytes:
    return file_pred_runner.run(f)


@svc.api(input=Image(), output=Image(mime_type="image/bmp"))
def echo_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    return np.array(f)  # type: ignore


@svc.api(
    input=Multipart(original=Image(), compared=Image()),
    output=Multipart(img1=Image(), img2=Image()),
)
def predict_multi_images(original: t.Dict[str, Image], compared: t.Dict[str, Image]):
    output_array = multi_ndarray_pred_runner.run_batch(
        np.array(original), np.array(compared)
    )
    img = fromarray(output_array)
    return dict(img1=img, img2=img)
