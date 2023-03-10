# pylint: disable=unused-argument
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from functools import partial

import numpy as np
import pytest

from bentoml.io import NumpyNdarray
from bentoml.exceptions import BadInput
from bentoml.exceptions import BentoMLException
from bentoml.grpc.utils import import_generated_stubs
from bentoml._internal.service.openapi.specification import Schema

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb, _ = import_generated_stubs()


class ExampleGeneric(str, np.generic):
    pass


example = np.zeros((2, 2, 3, 2))
from_example = NumpyNdarray.from_sample(example, enforce_dtype=True, enforce_shape=True)


def test_invalid_dtype():
    with pytest.raises(BentoMLException) as e:
        NumpyNdarray(dtype="asdf")
    assert "Invalid dtype" in str(e.value)

    generic = ExampleGeneric("asdf")
    with pytest.raises(BentoMLException) as e:
        _ = NumpyNdarray.from_sample(generic)  # type: ignore (test exception)
    assert "expects a 'numpy.array'" in str(e.value)


@pytest.mark.parametrize("dtype, expected", [("float", "number"), (">U8", "integer")])
def test_numpy_to_openapi_types(dtype: str, expected: str):
    assert NumpyNdarray(dtype=dtype)._openapi_types() == expected  # type: ignore (private functions warning)


def test_numpy_openapi_schema():
    nparray = NumpyNdarray().openapi_schema()
    assert nparray.type == "array"
    assert nparray.nullable
    assert nparray.items and nparray.items.type == "integer"

    ndarray = from_example.openapi_schema()
    assert nparray.type == "array"
    assert isinstance(nparray.items, Schema)
    items = ndarray.items
    assert items.type == "array"
    assert items.items and items.items.type == "number"


def test_numpy_openapi_request_body():
    nparray = NumpyNdarray().openapi_request_body()
    assert nparray["required"]

    assert nparray["content"]
    assert "application/json" in nparray["content"]

    ndarray = from_example.openapi_request_body()
    assert ndarray["required"]
    assert ndarray["content"]
    assert ndarray["content"]["application/json"].example == example.tolist()


def test_numpy_openapi_responses():
    responses = NumpyNdarray().openapi_responses()

    assert responses["content"]

    assert "application/json" in responses["content"]
    assert not responses["content"]["application/json"].example

    ndarray = from_example.openapi_request_body()
    assert ndarray["content"]
    assert ndarray["content"]["application/json"].example == example.tolist()


def test_numpy_openapi_example():
    r = NumpyNdarray().openapi_example()
    assert r is None

    r = from_example.openapi_example()
    assert r == example.tolist()

    nparray = NumpyNdarray(dtype="float")
    nparray.sample = ExampleGeneric("asdf")
    with pytest.raises(BadInput):
        nparray.openapi_example()


def test_verify_numpy_ndarray(caplog: LogCaptureFixture):
    partial_check = partial(from_example.validate_array, exception_cls=BentoMLException)

    with pytest.raises(BentoMLException) as ex:
        partial_check(np.array(["asdf"]))
    assert f'Expecting ndarray of dtype "{from_example._dtype}"' in str(ex.value)

    with pytest.raises(BentoMLException) as e:
        partial_check(np.array([[1]]))
    assert f'Expecting ndarray of shape "{from_example._shape}"' in str(e.value)

    # test cases where reshape is failed
    example = NumpyNdarray.from_sample(np.ones((2, 2, 3)))
    with caplog.at_level(logging.DEBUG):
        example.validate_array(np.array("asdf"))
    assert "Failed to reshape" in caplog.text


def test_from_sample_ensure_not_override():
    example = NumpyNdarray.from_sample(np.ones((2, 2, 3)), dtype=np.float32)
    assert example._dtype == np.float32

    example = NumpyNdarray.from_sample(np.ones((2, 2, 3)), shape=(2, 2, 3))
    assert example._shape == (2, 2, 3)


def generate_1d_array(dtype: pb.NDArray.DType.ValueType, length: int = 3):
    if dtype == pb.NDArray.DTYPE_BOOL:
        return [True] * length
    elif dtype == pb.NDArray.DTYPE_STRING:
        return ["a"] * length
    else:
        return [1] * length


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype",
    filter(lambda x: x > 0, [v.number for v in pb.NDArray.DType.DESCRIPTOR.values]),
)
async def test_from_proto(dtype: pb.NDArray.DType.ValueType) -> None:
    from bentoml._internal.io_descriptors.numpy import dtypepb_to_fieldpb_map
    from bentoml._internal.io_descriptors.numpy import dtypepb_to_npdtype_map

    np.testing.assert_array_equal(
        await NumpyNdarray(dtype=example.dtype, shape=example.shape).from_proto(
            example.ravel().tobytes(),
        ),
        example,
    )
    # DTYPE_UNSPECIFIED
    np.testing.assert_array_equal(
        await NumpyNdarray().from_proto(
            pb.NDArray(dtype=pb.NDArray.DType.DTYPE_UNSPECIFIED),
        ),
        np.empty(0),
    )
    np.testing.assert_array_equal(
        await NumpyNdarray().from_proto(
            pb.NDArray(shape=tuple(example.shape)),
        ),
        np.empty(tuple(example.shape)),
    )
    # different DTYPE
    np.testing.assert_array_equal(
        await NumpyNdarray().from_proto(
            pb.NDArray(
                dtype=dtype,
                **{dtypepb_to_fieldpb_map()[dtype]: generate_1d_array(dtype)},
            ),
        ),
        np.array(generate_1d_array(dtype), dtype=dtypepb_to_npdtype_map()[dtype]),
    )
    # given shape from message.
    np.testing.assert_array_equal(
        await NumpyNdarray().from_proto(
            pb.NDArray(shape=[3, 3], float_values=[1.0] * 9),
        ),
        np.array([[1.0] * 3] * 3),
    )


@pytest.mark.asyncio
async def test_exception_from_proto():
    with pytest.raises(BadInput):
        await NumpyNdarray().from_proto(pb.File(content=b"asdf"))  # type: ignore (testing exception)
        await NumpyNdarray().from_proto(b"asdf")
    with pytest.raises(BadInput) as exc_info:
        await NumpyNdarray().from_proto(pb.NDArray(dtype=123, string_values="asdf"))  # type: ignore (testing exception)
    assert "123 is invalid." == str(exc_info.value)
    with pytest.raises(BadInput) as exc_info:
        await NumpyNdarray().from_proto(
            pb.NDArray(string_values="asdf", float_values=[1.0, 2.0])
        )
    assert "Array contents can only be one of" in str(exc_info.value)


@pytest.mark.asyncio
async def test_exception_to_proto():
    with pytest.raises(BadInput):
        await NumpyNdarray(dtype=np.float32, enforce_dtype=True).to_proto(
            np.array("asdf")
        )
    with pytest.raises(BadInput):
        await NumpyNdarray(dtype=np.dtype(np.void)).to_proto(np.array("asdf"))


@pytest.mark.asyncio
async def test_to_proto() -> None:
    assert await NumpyNdarray().to_proto(example) == pb.NDArray(
        shape=example.shape,
        dtype=pb.NDArray.DType.DTYPE_DOUBLE,
        double_values=example.ravel().tolist(),
    )
