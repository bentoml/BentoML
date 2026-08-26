from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import pytest
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import wrappers_pb2

from _bentoml_impl.grpc.codec import decode_proto
from _bentoml_impl.grpc.codec import encode_proto
from _bentoml_impl.grpc.codec import proto_binding
from _bentoml_sdk.io_models import IODescriptor
from _bentoml_sdk.validators import TensorSchema
from bentoml.exceptions import InvalidArgument
from bentoml.grpc.utils import import_generated_stubs

pb, _ = import_generated_stubs("v1")

Array = t.Annotated[np.ndarray, TensorSchema("numpy-array")]


def _input_of(fn: t.Callable[..., t.Any]) -> type[IODescriptor]:
    return IODescriptor.from_input(fn, skip_self=True)


def _output_of(fn: t.Callable[..., t.Any]) -> type[IODescriptor]:
    return IODescriptor.from_output(fn)


def greet(self, name: str) -> str:
    return f"hello {name}"


def add(self, a: int, b: int) -> dict[str, int]:
    return {"sum": a + b}


def predict(self, x: Array) -> Array:
    return x * 2


def echo_file(self, data: Path) -> Path:
    return data


def test_proto_binding_maps_single_string_field_to_text():
    binding = proto_binding(_input_of(greet))
    assert binding.field == "text"
    assert binding.unwrap_key == "name"


def test_proto_binding_maps_string_return_to_text():
    binding = proto_binding(_output_of(greet))
    assert binding.field == "text"
    assert binding.unwrap_key is None


def test_proto_binding_maps_multi_field_input_to_json():
    binding = proto_binding(_input_of(add))
    assert binding.field == "json"
    assert binding.unwrap_key is None


def test_proto_binding_maps_tensor_field_to_ndarray():
    binding = proto_binding(_input_of(predict))
    assert binding.field == "ndarray"
    assert binding.unwrap_key == "x"


def test_proto_binding_maps_path_field_to_file():
    binding = proto_binding(_input_of(echo_file))
    assert binding.field == "file"
    assert binding.unwrap_key == "data"


@pytest.mark.asyncio
async def test_text_round_trip():
    spec = _input_of(greet)
    encoded_field, encoded = await encode_proto(spec, {"name": "willow"})
    assert encoded_field == "text"
    assert encoded.value == "willow"

    decoded = await decode_proto(spec, "text", encoded)
    assert decoded.name == "willow"


@pytest.mark.asyncio
async def test_json_round_trip():
    spec = _input_of(add)
    encoded_field, encoded = await encode_proto(spec, {"a": 2, "b": 3})
    assert encoded_field == "json"
    parsed = json_format.MessageToDict(encoded)
    assert parsed == {"a": 2, "b": 3}

    decoded = await decode_proto(spec, "json", encoded)
    assert decoded.a == 2
    assert decoded.b == 3


@pytest.mark.asyncio
async def test_ndarray_round_trip():
    spec = _input_of(predict)
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    encoded_field, encoded = await encode_proto(spec, array)
    assert encoded_field == "ndarray"
    assert list(encoded.shape) == [2, 2]

    decoded = await decode_proto(spec, "ndarray", encoded)
    np.testing.assert_array_equal(decoded.x, array)


@pytest.mark.asyncio
async def test_file_round_trip(tmp_path: Path):
    spec = _input_of(echo_file)
    payload = b"grpc-file-bytes"
    source = tmp_path / "input.bin"
    source.write_bytes(payload)

    encoded_field, encoded = await encode_proto(spec, source)
    assert encoded_field == "file"
    assert encoded.content == payload

    decoded = await decode_proto(spec, "file", encoded)
    assert Path(decoded.data).read_bytes() == payload


@pytest.mark.asyncio
async def test_output_string_encodes_as_text():
    spec = _output_of(greet)
    field, encoded = await encode_proto(spec, "hello willow")
    assert field == "text"
    assert isinstance(encoded, wrappers_pb2.StringValue)
    assert encoded.value == "hello willow"


@pytest.mark.asyncio
async def test_rejects_serialized_bytes():
    spec = _input_of(greet)
    with pytest.raises(InvalidArgument, match="serialized_bytes"):
        await decode_proto(spec, "serialized_bytes", b"nope")


@pytest.mark.asyncio
async def test_rejects_mismatched_proto_field():
    spec = _input_of(greet)
    with pytest.raises(InvalidArgument, match="text"):
        await decode_proto(spec, "json", struct_pb2.Value())
