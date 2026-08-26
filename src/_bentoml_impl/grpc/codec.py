from __future__ import annotations

import os
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import wrappers_pb2

from _bentoml_sdk.io_models import IODescriptor
from _bentoml_sdk.io_models import IORootModel
from bentoml.exceptions import InvalidArgument
from bentoml.grpc.utils import import_generated_stubs

if t.TYPE_CHECKING:
    from google.protobuf.message import Message

    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb, _ = import_generated_stubs("v1")

_SUPPORTED_FIELDS = ("text", "json", "ndarray", "file")


@dataclass(frozen=True)
class ProtoBinding:
    """How a pydantic IO spec maps onto the v1 Request/Response oneof."""

    field: str
    unwrap_key: str | None = None


def _resolve_ref(schema: dict[str, t.Any], root: dict[str, t.Any]) -> dict[str, t.Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/$defs/"):
        return schema
    name = ref.rsplit("/", 1)[-1]
    resolved = (root.get("$defs") or {}).get(name)
    return resolved if isinstance(resolved, dict) else schema


def _field_from_schema(schema: dict[str, t.Any]) -> str:
    type_ = schema.get("type")
    fmt = schema.get("format")
    if type_ == "tensor":
        return "ndarray"
    if type_ == "file":
        return "file"
    if type_ == "string" and fmt in {"binary", "byte"}:
        return "file"
    if type_ == "string":
        return "text"
    if type_ == "dataframe":
        raise InvalidArgument(
            "pandas DataFrame IO is not supported over gRPC for @bentoml.service() yet"
        )
    return "json"


def proto_binding(spec: type[IODescriptor]) -> ProtoBinding:
    """Infer the v1 proto oneof field for an IO spec."""
    schema = spec.model_json_schema()
    if issubclass(spec, IORootModel):
        return ProtoBinding(_field_from_schema(schema), None)

    props = schema.get("properties") or {}
    if schema.get("type") == "object" and len(props) == 1:
        key, child = next(iter(props.items()))
        child = _resolve_ref(child, schema)
        field = _field_from_schema(child)
        if field != "json":
            return ProtoBinding(field, key)
    return ProtoBinding(_field_from_schema(schema), None)


def _unwrap_value(spec: type[IODescriptor], binding: ProtoBinding, obj: t.Any) -> t.Any:
    if issubclass(spec, IORootModel) and isinstance(obj, IORootModel):
        return obj.root
    if binding.unwrap_key is None:
        if hasattr(obj, "model_dump") and not isinstance(
            obj, (bytes, str, Path, dict, list)
        ):
            dump = obj.model_dump()
            if isinstance(dump, dict):
                return dump
        return obj
    key = binding.unwrap_key
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj


def _wrap_value(
    spec: type[IODescriptor], binding: ProtoBinding, value: t.Any
) -> IODescriptor:
    if issubclass(spec, IORootModel):
        return spec.from_inputs(value)
    if binding.unwrap_key is not None:
        return spec.model_validate({binding.unwrap_key: value})
    if isinstance(value, spec):
        return value
    if isinstance(value, dict):
        return spec.model_validate(value)
    return spec.from_inputs(value)


async def _ndarray_from_proto(field: t.Any) -> t.Any:
    from bentoml._internal.io_descriptors.numpy import NumpyNdarray

    return await NumpyNdarray().from_proto(field)


async def _ndarray_to_proto(obj: t.Any) -> Message:
    from bentoml._internal.io_descriptors.numpy import NumpyNdarray

    return await NumpyNdarray().to_proto(obj)


def _file_from_proto(field: t.Any) -> bytes:
    if isinstance(field, bytes):
        return field
    content = getattr(field, "content", None)
    if content is None:
        raise InvalidArgument("File proto is missing content")
    return bytes(content)


def _bytes_to_path(body: bytes) -> Path:
    # FileSchema.decode() looks up the HTTP request temp dir; gRPC has none,
    # so persist bytes to a local tempfile the validator can treat as a Path.
    fd, name = tempfile.mkstemp(prefix="bentoml-grpc-")
    with os.fdopen(fd, "wb") as handle:
        handle.write(body)
    return Path(name)


def _file_to_proto(obj: t.Any) -> pb.File:
    if isinstance(obj, bytes):
        body = obj
        kind = "application/octet-stream"
    elif isinstance(obj, (str, Path)):
        path = Path(obj)
        body = path.read_bytes()
        kind = "application/octet-stream"
    elif hasattr(obj, "read"):
        body = obj.read()
        if isinstance(body, str):
            body = body.encode("utf-8")
        kind = "application/octet-stream"
    else:
        raise InvalidArgument(f"Cannot encode {type(obj)!r} as a gRPC file")
    return pb.File(kind=kind, content=body)


def _json_from_proto(field: t.Any) -> t.Any:
    if isinstance(field, bytes):
        import json

        return json.loads(field)
    return json_format.MessageToDict(field, preserving_proto_field_name=True)


def _json_to_proto(obj: t.Any) -> struct_pb2.Value:
    msg = struct_pb2.Value()
    if obj is None:
        return msg
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    json_format.ParseDict(obj, msg)
    return msg


async def decode_proto(
    spec: type[IODescriptor], field_name: str | None, value: t.Any
) -> IODescriptor:
    """Decode a Request/Response oneof value into an IODescriptor instance."""
    if field_name == "serialized_bytes":
        raise InvalidArgument(
            "serialized_bytes / pickle payloads are not supported over gRPC "
            "for @bentoml.service()"
        )
    binding = proto_binding(spec)
    if field_name is None:
        raise InvalidArgument("gRPC request is missing a content field")
    if field_name not in _SUPPORTED_FIELDS:
        raise InvalidArgument(
            f"Unsupported gRPC content field {field_name!r}; "
            f"accepted fields: {', '.join(_SUPPORTED_FIELDS)}"
        )
    if field_name != binding.field:
        raise InvalidArgument(
            f"{spec.__name__} expects gRPC field {binding.field!r}, got {field_name!r}"
        )

    if binding.field == "text":
        if isinstance(value, bytes):
            decoded: t.Any = value.decode("utf-8")
        else:
            decoded = value.value if hasattr(value, "value") else str(value)
    elif binding.field == "json":
        decoded = _json_from_proto(value)
    elif binding.field == "ndarray":
        decoded = await _ndarray_from_proto(value)
    else:
        decoded = _bytes_to_path(_file_from_proto(value))
    return _wrap_value(spec, binding, decoded)


async def encode_proto(spec: type[IODescriptor], obj: t.Any) -> tuple[str, t.Any]:
    """Encode a Python value to (oneof field name, proto message)."""
    binding = proto_binding(spec)
    value = _unwrap_value(spec, binding, obj)
    if binding.field == "text":
        return binding.field, wrappers_pb2.StringValue(value=str(value))
    if binding.field == "json":
        return binding.field, _json_to_proto(value)
    if binding.field == "ndarray":
        return binding.field, await _ndarray_to_proto(value)
    return binding.field, _file_to_proto(value)
