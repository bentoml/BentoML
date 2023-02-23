from __future__ import annotations

import time
import typing as t
from typing import TYPE_CHECKING

from pydantic import BaseModel
from context_server_interceptor import AsyncContextInterceptor

import bentoml
from bentoml.io import File
from bentoml.io import JSON
from bentoml.io import Text
from bentoml.io import Image
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray
from bentoml.io import PandasSeries
from bentoml.io import PandasDataFrame
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils.metrics import exponential_buckets

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import PIL.Image
    from numpy.typing import NDArray

    from bentoml._internal.types import FileLike
    from bentoml._internal.types import JSONSerializable
    from bentoml.picklable_model import get_runnable
    from bentoml._internal.runner.runner import RunnerMethod

    RunnableImpl = get_runnable(bentoml.picklable_model.get("py_model.case-1.grpc.e2e"))

    class PythonModelRunner(bentoml.Runner):
        predict_file: RunnerMethod[RunnableImpl, [list[FileLike[bytes]]], list[bytes]]
        echo_json: RunnerMethod[
            RunnableImpl, [list[JSONSerializable]], list[JSONSerializable]
        ]
        echo_ndarray: RunnerMethod[RunnableImpl, [NDArray[t.Any]], NDArray[t.Any]]
        double_ndarray: RunnerMethod[RunnableImpl, [NDArray[t.Any]], NDArray[t.Any]]
        multiply_float_ndarray: RunnerMethod[
            RunnableImpl,
            [NDArray[np.float32], NDArray[np.float32]],
            NDArray[np.float32],
        ]
        double_dataframe_column: RunnerMethod[
            RunnableImpl, [pd.DataFrame], pd.DataFrame
        ]
        echo_dataframe: RunnerMethod[RunnableImpl, [pd.DataFrame], pd.DataFrame]

else:
    np = LazyLoader("np", globals(), "numpy")
    pd = LazyLoader("pd", globals(), "pandas")
    PIL = LazyLoader("PIL", globals(), "PIL")
    PIL.Image = LazyLoader("PIL.Image", globals(), "PIL.Image")


py_model = t.cast(
    "PythonModelRunner",
    bentoml.picklable_model.get("py_model.case-1.grpc.e2e").to_runner(),
)

svc = bentoml.Service(name="general_grpc_service.case-1.e2e", runners=[py_model])

svc.add_grpc_interceptor(AsyncContextInterceptor, usage="NLP", accuracy_score=0.8247)


class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float


class IrisClassificationRequest(BaseModel):
    request_id: str
    iris_features: IrisFeatures


@svc.api(input=Text(), output=Text())
async def bonjour(inp: str) -> str:
    return f"Hello, {inp}!"


@svc.api(input=JSON(), output=JSON())
async def echo_json(json_obj: JSONSerializable) -> JSONSerializable:
    batched = await py_model.echo_json.async_run([json_obj])
    return batched[0]


@svc.api(
    input=JSON(pydantic_model=IrisClassificationRequest),
    output=JSON(),
)
def echo_json_validate(input_data: IrisClassificationRequest) -> dict[str, float]:
    print("request_id: ", input_data.request_id)
    return input_data.iris_features.dict()


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def double_ndarray(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    return await py_model.double_ndarray.async_run(arr)


@svc.api(input=NumpyNdarray.from_sample(np.random.rand(2, 2)), output=NumpyNdarray())
async def echo_ndarray_from_sample(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    assert arr.shape == (2, 2)
    return await py_model.echo_ndarray.async_run(arr)


@svc.api(input=NumpyNdarray(shape=(2, 2), enforce_shape=True), output=NumpyNdarray())
async def echo_ndarray_enforce_shape(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    assert arr.shape == (2, 2)
    return await py_model.echo_ndarray.async_run(arr)


@svc.api(
    input=NumpyNdarray(dtype=np.float32, enforce_dtype=True), output=NumpyNdarray()
)
async def echo_ndarray_enforce_dtype(arr: NDArray[t.Any]) -> NDArray[t.Any]:
    assert arr.dtype == np.float32
    return await py_model.echo_ndarray.async_run(arr)


@svc.api(input=PandasDataFrame(orient="columns"), output=PandasDataFrame())
async def echo_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    return df


@svc.api(
    input=PandasDataFrame.from_sample(
        pd.DataFrame({"age": [3, 29], "height": [94, 170], "weight": [31, 115]}),
        orient="columns",
    ),
    output=PandasDataFrame(),
)
async def echo_dataframe_from_sample(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    return df


@svc.api(input=PandasSeries.from_sample(pd.Series([1, 2, 3])), output=PandasSeries())
async def echo_series_from_sample(series: pd.Series) -> pd.Series:
    assert isinstance(series, pd.Series)
    return series


@svc.api(
    input=PandasDataFrame(dtype={"col1": "int64"}, orient="columns"),
    output=PandasDataFrame(),
)
async def double_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    assert df["col1"].dtype == "int64"
    output = await py_model.double_dataframe_column.async_run(df)
    dfo = pd.DataFrame()
    dfo["col1"] = output
    return dfo


@svc.api(input=File(), output=File())
async def predict_file(f: FileLike[bytes]) -> bytes:
    batch_ret = await py_model.predict_file.async_run([f])
    return batch_ret[0]


@svc.api(input=Image(mime_type="image/bmp"), output=Image(mime_type="image/bmp"))
async def echo_image(f: PIL.Image.Image) -> NDArray[t.Any]:
    assert isinstance(f, PIL.Image.Image)
    return np.array(f)


histogram = bentoml.metrics.Histogram(
    name="inference_latency",
    documentation="Inference latency in seconds",
    labelnames=["model_name", "model_version"],
    buckets=exponential_buckets(0.001, 1.5, 10.0),
)


@svc.api(
    input=Multipart(
        original=Image(mime_type="image/bmp"), compared=Image(mime_type="image/bmp")
    ),
    output=Multipart(meta=Text(), result=Image(mime_type="image/bmp")),
)
async def predict_multi_images(original: Image, compared: Image):
    start = time.perf_counter()
    output_array = await py_model.multiply_float_ndarray.async_run(
        np.array(original), np.array(compared)
    )
    histogram.labels(model_name=py_model.name, model_version="v1").observe(
        time.perf_counter() - start
    )
    img = PIL.Image.fromarray(output_array)
    return {"meta": "success", "result": img}


@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
def ensure_metrics_are_registered(_: str) -> None:
    histograms = [
        m.name
        for m in bentoml.metrics.text_string_to_metric_families()
        if m.type == "histogram"
    ]
    assert "inference_latency" in histograms
