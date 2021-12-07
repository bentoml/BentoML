import typing as t

import numpy as np
import pandas as pd
import pydantic
from PIL.Image import Image as PILImage
from PIL.Image import fromarray

import bentoml
import bentoml.sklearn
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
    name: str
    endpoints: t.List[str]


class PickleModel:
    @staticmethod
    def predict_file(input_files: t.List[FileLike]) -> t.List[bytes]:
        return [f.read() for f in input_files]

    @staticmethod
    def echo_json(input_datas: JSONSerializable) -> JSONSerializable:
        return input_datas

    @staticmethod
    def echo_multi_ndarray(
        *input_arr: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> t.Tuple["np.ndarray[t.Any, np.dtype[t.Any]]", ...]:
        return input_arr

    @staticmethod
    def predict_ndarray(
        arr: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        assert isinstance(arr, np.ndarray)
        return arr * 2

    @staticmethod
    def predict_multi_ndarray(
        arr1: "np.ndarray[t.Any, np.dtype[t.Any]]",
        arr2: "np.ndarray[t.Any, np.dtype[t.Any]]",
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        assert isinstance(arr1, np.ndarray)
        assert isinstance(arr2, np.ndarray)
        return (arr1 + arr2) // 2

    @staticmethod
    def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)
        output = df[["col1"]] * 2
        assert isinstance(output, pd.DataFrame)
        return output


bentoml.sklearn.save("sk_model", PickleModel())


json_echo_runner = bentoml.sklearn.load_runner("sk_model", function_name="echo_json")
ndarray_pred_runner = bentoml.sklearn.load_runner(
    "sk_model", function_name="predict_ndarray"
)
dataframe_pred_runner = bentoml.sklearn.load_runner(
    "sk_model", function_name="predict_dataframe"
)
file_pred_runner = bentoml.sklearn.load_runner("sk_model", function_name="predict_file")

multi_ndarray_pred_runner = bentoml.sklearn.load_runner(
    "sk_model", function_name="predict_multi_ndarray"
)
echo_multi_ndarray_pred_runner = bentoml.sklearn.load_runner(
    "sk_model", function_name="echo_multi_ndarray"
)


svc = bentoml.Service(
    name="general",
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
async def echo_json(json_obj: JSONSerializable) -> JSONSerializable:
    return await json_echo_runner.async_run(json_obj)


@svc.api(
    input=JSON(pydantic_model=_Schema(name="test", endpoints=["predict", "health"])),
    output=JSON(),
)
async def pydantic_json(json_obj: JSONSerializable) -> JSONSerializable:
    return await json_echo_runner.async_run(json_obj)


@svc.api(
    input=NumpyNdarray(shape=(2, 2), enforce_shape=True),
    output=NumpyNdarray(shape=(1, 4)),
)
async def predict_ndarray_enforce_shape(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.shape == (2, 2)
    return await ndarray_pred_runner.async_run(inp)


@svc.api(
    input=NumpyNdarray(dtype="uint8", enforce_dtype=True),
    output=NumpyNdarray(dtype="str"),
)
async def predict_ndarray_enforce_dtype(
    inp: "np.ndarray[t.Any, np.dtype[t.Any]]",
) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert inp.dtype == np.dtype("uint8")
    return await ndarray_pred_runner.async_run(inp)


@svc.api(
    input=PandasDataFrame(dtype={"col1": "int64"}, orient="records"),
    output=PandasSeries(),
)
async def predict_dataframe(df: "pd.DataFrame") -> "pd.Series":
    assert df["col1"].dtype == "int64"
    output = await dataframe_pred_runner.async_run(df)
    assert isinstance(output, pd.Series)
    return output


@svc.api(input=File(), output=File())
async def predict_file(f: FileLike) -> bytes:
    return await file_pred_runner.async_run(f)


@svc.api(input=Image(), output=Image(mime_type="image/bmp"))
async def echo_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    return np.array(f)


@svc.api(
    input=Multipart(original=Image(), compared=Image()),
    output=Multipart(img1=Image(), img2=Image()),
)
async def predict_multi_images(
    original: t.Dict[str, Image], compared: t.Dict[str, Image]
):
    output_array = await multi_ndarray_pred_runner.async_run_batch(
        np.array(original), np.array(compared)
    )
    img = fromarray(output_array)
    return dict(img1=img, img2=img)
