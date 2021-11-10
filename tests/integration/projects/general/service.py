import typing as t

import numpy as np
import pandas as pd
import pydantic
from PIL import Image as PILImage

import bentoml.sklearn
from bentoml._internal.types import FileLike, JSONSerializable
from bentoml.io import (
    JSON,
    File,
    Image,
    Multipart,
    NumpyNdarray,
    PandasDataFrame,
    PandasSeries,
)


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
    def echo_multi_ndarray(*input_arr: np.ndarray) -> t.Tuple[np.ndarray, ...]:
        return input_arr

    @staticmethod
    def predict_ndarray(input_arr: np.ndarray) -> np.ndarray:
        return input_arr * 2

    @staticmethod
    def predict_multi_ndarray(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        return (arr1 + arr2) // 2

    @staticmethod
    def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)
        output = df[["col1"]] * 2
        assert isinstance(output, pd.DataFrame)
        return output


bentoml.sklearn.save("sk_model", PickleModel())


json_pred_runner = bentoml.sklearn.load_runner("sk_model", function_name="echo_json")
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
        json_pred_runner,
        ndarray_pred_runner,
        dataframe_pred_runner,
        file_pred_runner,
        multi_ndarray_pred_runner,
        echo_multi_ndarray_pred_runner,
    ],
)


@svc.api(input=JSON(), output=JSON())
def echo_json(json_obj: JSONSerializable) -> JSONSerializable:
    return json_pred_runner.run(json_obj)


@svc.api(
    input=JSON(pydantic_model=_Schema(name="test", endpoints=["predict", "health"])),
    output=JSON(),
)
def pydantic_json(json_obj: JSONSerializable) -> JSONSerializable:
    return json_pred_runner.run(json_obj)


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_np_array(inp: "np.ndarray"):
    return inp * 2


@svc.api(input=JSON(), output=JSON())
def predict_array(json_obj: JSONSerializable) -> JSONSerializable:
    array = np.array(json_obj)
    array_out = json_pred_runner.run(array)
    return array_out.tolist()


@svc.api(
    input=PandasDataFrame(dtype={"col1": "int64"}, orient="records"),
    output=PandasSeries(),
)
def predict_dataframe(df):
    assert df["col1"].dtype == np.int64
    output = dataframe_pred_runner.run(df)
    assert isinstance(output, pd.Series)
    return output


@svc.api(input=File(), output=File())
def predict_file(f):
    return file_pred_runner.run(f)


@svc.api(input=Image(), output=Image())
def echo_image(f):
    return np.array(f)


@svc.api(input=File(), output=File())
def predict_invalid_filetype(f):
    return 1


@svc.api(input=Image(), output=Image())
def predict_invalid_imgtype(f):
    return 1


@svc.api(input=Multipart(original=Image(), compared=Image()), output=Image())
def predict_multi_images(original, compared):
    output_array = multi_ndarray_pred_runner.run_batch(
        np.array(original), np.array(compared)
    )
    return PILImage.fromarray(output_array)


@svc.api(
    input=Multipart(original=Image(), compared=Image()),
    output=Multipart(original=Image(), compared=Image()),
)
def echo_return_multipart(original, compared):
    res = echo_multi_ndarray_pred_runner.run_batch(
        np.array(original), np.array(compared)
    )
    return dict(original=res[0], compared=res[1])
