import typing as t

import numpy as np
import pandas as pd
from PIL import Image as PILImage

import bentoml.sklearn
from bentoml._internal.io_descriptors.pandas import PandasDataFrame, PandasSeries
from bentoml._internal.types import FileLike, JSONSerializable
from bentoml.io import JSON, File, Image, Multipart


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


"""
from typing import Sequence

from packaging import version
from bentoml import __version__ as BENTOML_VERSION
@svc.api(DataframeHandler, dtype={"col1": "int"}, batch=True)  # deprecated
def predict_dataframe_v1(df):
    return self.artifacts.model.predict_dataframe(df)


@svc.api(input=MultiImageInput(input_names=('original', 'compared')), batch=True)
def predict_multi_images(originals, compareds):
    return self.artifacts.model.predict_multi_images(originals, compareds)


@svc.api(input=ImageInput(), batch=True)
def predict_image(images):
    return self.artifacts.model.predict_image(images)


@svc.api(
    input=JsonInput(),
    mb_max_latency=1000,
    mb_max_batch_size=2000,
    batch=True,
)
def predict_with_sklearn(jsons):
    return self.artifacts.sk_model.predict(jsons)


@svc.api(input=FileInput(), batch=True)
def predict_file(files):
    return self.artifacts.model.predict_file(files)


@svc.api(input=JsonInput(), batch=True)
def echo_json(input_datas):
    return self.artifacts.model.echo_json(input_datas)


CUSTOM_ROUTE = "$~!@%^&*()_-+=[]\\|;:,./predict"


@svc.api(
    route=CUSTOM_ROUTE,
    input=JsonInput(),
    batch=True,
)
def customezed_route(input_datas):
    return input_datas


CUSTOM_SCHEMA = {
    "application/json": {
        "schema": {
            "type": "object",
            "required": ["field1", "field2"],
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "uuid"},
            },
        },
    }
}


@svc.api(input=JsonInput(request_schema=CUSTOM_SCHEMA), batch=True)
def customezed_schema(input_datas):
    return input_datas


@svc.api(input=JsonInput(), batch=True)
def predict_strict_json(input_datas, tasks: Sequence[InferenceTask] = None):
    filtered_jsons = []
    for j, t in zip(input_datas, tasks):
        if t.http_headers.content_type != "application/json":
            t.discard(http_status=400, err_msg="application/json only")
        else:
            filtered_jsons.append(j)
    return self.artifacts.model.echo_json(filtered_jsons)


@svc.api(input=JsonInput(), batch=True)
def predict_direct_json(input_datas, tasks: Sequence[InferenceTask] = None):
    filtered_jsons = []
    for j, t in zip(input_datas, tasks):
        if t.http_headers.content_type != "application/json":
            t.discard(http_status=400, err_msg="application/json only")
        else:
            filtered_jsons.append(j)
    rets = self.artifacts.model.echo_json(filtered_jsons)
    return [
        InferenceResult(http_status=200, data=json.dumps(result)) for result in rets
    ]


@svc.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
def echo_with_delay(input_datas):
    data = input_datas[0]
    time.sleep(data['b'] + data['a'] * len(input_datas))
    return input_datas


@svc.api(input=JsonInput(), mb_max_latency=10000 * 1000, batch=True)
def echo_batch_size(input_datas=10):
    data = input_datas[0]
    time.sleep(data['b'] + data['a'] * len(input_datas))
    batch_size = len(input_datas)
    return [batch_size] * batch_size


@svc.api(input=JsonInput())
def echo_json(input_data):
    return input_data


if version.parse(BENTOML_VERSION) > version.parse("0.12.1+0"):

    @svc.api(input=JsonInput(), output=JsonOutput(ensure_ascii=True))
    def echo_json_ensure_ascii(input_data):
        return input_data
"""
