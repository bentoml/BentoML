from __future__ import annotations

import io
import random
import traceback
from typing import TYPE_CHECKING
from functools import partial

import pytest

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import import_generated_stubs
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call
from bentoml.testing.grpc import randomize_pb_ndarray
from bentoml._internal.types import LazyType
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import grpc
    import numpy as np
    import pandas as pd
    import PIL.Image as PILImage
    from google.protobuf import struct_pb2
    from google.protobuf import wrappers_pb2

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml._internal import external_typing as ext
else:
    pb, _ = import_generated_stubs()
    grpc, _ = import_grpc()
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")
    np = LazyLoader("np", globals(), "numpy")
    pd = LazyLoader("pd", globals(), "pandas")
    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


def assert_ndarray(
    resp: pb.Response,
    assert_shape: list[int],
    assert_dtype: pb.NDArray.DType.ValueType,
) -> bool:
    # Hide traceback from pytest
    __tracebackhide__ = True  # pylint: disable=unused-variable

    dtype = resp.ndarray.dtype
    try:
        assert resp.ndarray.shape == assert_shape
        assert dtype == assert_dtype
        return True
    except AssertionError:
        traceback.print_exc()
        return False


def make_iris_proto(**fields: struct_pb2.Value) -> struct_pb2.Value:
    return struct_pb2.Value(
        struct_value=struct_pb2.Struct(
            fields={
                "request_id": struct_pb2.Value(string_value="123"),
                "iris_features": struct_pb2.Value(
                    struct_value=struct_pb2.Struct(fields=fields)
                ),
            }
        )
    )


@pytest.mark.asyncio
async def test_numpy(host: str):
    async with create_channel(host) as channel:
        await async_client_call(
            "double_ndarray",
            channel=channel,
            data={"ndarray": randomize_pb_ndarray((1000,))},
            assert_data=partial(
                assert_ndarray, assert_shape=[1000], assert_dtype=pb.NDArray.DTYPE_FLOAT
            ),
        )
        await async_client_call(
            "double_ndarray",
            channel=channel,
            data={"ndarray": pb.NDArray(shape=[2, 2], int32_values=[1, 2, 3, 4])},
            assert_data=lambda resp: resp.ndarray.int32_values == [2, 4, 6, 8],
        )
        await async_client_call(
            "double_ndarray",
            channel=channel,
            data={"ndarray": pb.NDArray(string_values=np.array(["2", "2f"]))},
            assert_code=grpc.StatusCode.INTERNAL,
        )
        await async_client_call(
            "double_ndarray",
            channel=channel,
            data={
                "ndarray": pb.NDArray(
                    dtype=123, string_values=np.array(["2", "2f"])  # type: ignore (test exception)
                )
            },
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "double_ndarray",
            channel=channel,
            data={"serialized_bytes": np.array([1, 2, 3, 4]).ravel().tobytes()},
        )
        await async_client_call(
            "double_ndarray",
            channel=channel,
            data={"text": wrappers_pb2.StringValue(value="asdf")},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "echo_ndarray_enforce_shape",
            channel=channel,
            data={"ndarray": randomize_pb_ndarray((1000,))},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "echo_ndarray_enforce_dtype",
            channel=channel,
            data={"ndarray": pb.NDArray(string_values=np.array(["2", "2f"]))},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )


@pytest.mark.asyncio
async def test_json(host: str):
    async with create_channel(host) as channel:
        await async_client_call(
            "echo_json",
            channel=channel,
            data={"json": struct_pb2.Value(string_value='"hi"')},
            assert_data=lambda resp: resp.json.string_value == '"hi"',
        )
        await async_client_call(
            "echo_json",
            channel=channel,
            data={
                "serialized_bytes": b'{"request_id": "123", "iris_features": {"sepal_len":2.34,"sepal_width":1.58, "petal_len":6.52, "petal_width":3.23}}'
            },
            assert_data=lambda resp: resp.json  # type: ignore (bad lambda types)
            == make_iris_proto(
                sepal_len=struct_pb2.Value(number_value=2.34),
                sepal_width=struct_pb2.Value(number_value=1.58),
                petal_len=struct_pb2.Value(number_value=6.52),
                petal_width=struct_pb2.Value(number_value=3.23),
            ),
        )
        await async_client_call(
            "echo_json_validate",
            channel=channel,
            data={
                "json": make_iris_proto(
                    **{
                        k: struct_pb2.Value(number_value=random.uniform(1.0, 6.0))
                        for k in [
                            "sepal_len",
                            "sepal_width",
                            "petal_len",
                            "petal_width",
                        ]
                    }
                )
            },
        )
        await async_client_call(
            "echo_json",
            channel=channel,
            data={"serialized_bytes": b"\n?xfa"},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "echo_json",
            channel=channel,
            data={"text": wrappers_pb2.StringValue(value="asdf")},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "echo_json_validate",
            channel=channel,
            data={
                "json": make_iris_proto(
                    sepal_len=struct_pb2.Value(number_value=2.34),
                    sepal_width=struct_pb2.Value(number_value=1.58),
                    petal_len=struct_pb2.Value(number_value=6.52),
                ),
            },
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )


@pytest.mark.asyncio
async def test_file(host: str, bin_file: str):
    # Test File as binary
    with open(str(bin_file), "rb") as f:
        fb = f.read()

    async with create_channel(host) as channel:
        await async_client_call(
            "predict_file",
            channel=channel,
            data={"serialized_bytes": fb},
            assert_data=lambda resp: resp.file.content == fb,
        )
        await async_client_call(
            "predict_file",
            channel=channel,
            data={"file": pb.File(kind="application/octet-stream", content=fb)},
            assert_data=lambda resp: resp.file.content == b"\x810\x899"
            and resp.file.kind == "application/octet-stream",
        )
        await async_client_call(
            "predict_file",
            channel=channel,
            data={"file": pb.File(kind="application/pdf", content=fb)},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "predict_file",
            channel=channel,
            data={"text": wrappers_pb2.StringValue(value="asdf")},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )


def assert_image(
    resp: pb.Response | pb.Part,
    assert_kind: str,
    im_file: str | ext.NpNDArray,
) -> bool:
    fio = io.BytesIO(resp.file.content)
    fio.name = "test.bmp"
    img = PILImage.open(fio)
    a1 = np.array(img)
    if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(im_file):
        a2 = PILImage.fromarray(im_file)
    else:
        assert isinstance(im_file, str)
        a2 = PILImage.open(im_file)
    try:
        assert resp.file.kind == assert_kind
        np.testing.assert_array_almost_equal(a1, np.array(a2))
        return True
    except AssertionError:
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_image(host: str, img_file: str):
    with open(str(img_file), "rb") as f:
        fb = f.read()

    async with create_channel(host) as channel:
        await async_client_call(
            "echo_image",
            channel=channel,
            data={"serialized_bytes": fb},
            assert_data=partial(
                assert_image, im_file=img_file, assert_kind="image/bmp"
            ),
        )
        await async_client_call(
            "echo_image",
            channel=channel,
            data={"file": pb.File(kind="image/bmp", content=fb)},
            assert_data=partial(
                assert_image, im_file=img_file, assert_kind="image/bmp"
            ),
        )
        await async_client_call(
            "echo_image",
            channel=channel,
            data={"file": pb.File(kind="application/pdf", content=fb)},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )
        await async_client_call(
            "echo_image",
            channel=channel,
            data={"text": wrappers_pb2.StringValue(value="asdf")},
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )


@pytest.mark.asyncio
async def test_pandas(host: str):
    async with create_channel(host) as channel:
        await async_client_call(
            "echo_dataframe",
            channel=channel,
            data={
                "dataframe": pb.DataFrame(
                    column_names=[
                        str(i) for i in pd.RangeIndex(0, 3, 1, dtype=np.int64).tolist()
                    ],
                    columns=[
                        pb.Series(int32_values=[1]),
                        pb.Series(int32_values=[2]),
                        pb.Series(int32_values=[3]),
                    ],
                ),
            },
        )
        await async_client_call(
            "echo_dataframe_from_sample",
            channel=channel,
            data={
                "dataframe": pb.DataFrame(
                    column_names=["age", "height", "weight"],
                    columns=[
                        pb.Series(int64_values=[12, 23]),
                        pb.Series(int64_values=[40, 83]),
                        pb.Series(int64_values=[32, 89]),
                    ],
                ),
            },
        )
        await async_client_call(
            "double_dataframe",
            channel=channel,
            data={
                "dataframe": pb.DataFrame(
                    column_names=["col1"],
                    columns=[pb.Series(int64_values=[23])],
                ),
            },
            assert_data=lambda resp: resp.dataframe  # type: ignore (bad lambda types)
            == pb.DataFrame(
                column_names=["col1"],
                columns=[pb.Series(int64_values=[46])],
            ),
        )
        await async_client_call(
            "echo_dataframe",
            channel=channel,
            data={
                "dataframe": pb.DataFrame(
                    column_names=["col1"],
                    columns=[pb.Series(int64_values=[23], int32_values=[23])],
                ),
            },
            assert_code=grpc.StatusCode.INVALID_ARGUMENT,
        )


@pytest.mark.asyncio
async def test_pandas_series(host: str):
    async with create_channel(host) as channel:
        await async_client_call(
            "echo_series_from_sample",
            channel=channel,
            data={"series": pb.Series(float_values=[1.0, 2.0, 3.0])},
            assert_data=lambda resp: resp.series
            == pb.Series(float_values=[1.0, 2.0, 3.0]),
        )


def assert_multi_images(resp: pb.Response, method: str, im_file: str) -> bool:
    assert method == "pred_multi_images"
    img = PILImage.open(im_file)
    arr = np.array(img)
    expected = arr * arr
    return assert_image(
        resp.multipart.fields["result"],
        assert_kind="image/bmp",
        im_file=expected,
    )


@pytest.mark.asyncio
async def test_multipart(host: str, img_file: str):
    with open(str(img_file), "rb") as f:
        fb = f.read()

    async with create_channel(host) as channel:
        await async_client_call(
            "predict_multi_images",
            channel=channel,
            data={
                "multipart": {
                    "fields": {
                        "original": pb.Part(file=pb.File(kind="image/bmp", content=fb)),
                        "compared": pb.Part(file=pb.File(kind="image/bmp", content=fb)),
                    }
                }
            },
            assert_data=partial(
                assert_multi_images, method="pred_multi_images", im_file=img_file
            ),
        )
