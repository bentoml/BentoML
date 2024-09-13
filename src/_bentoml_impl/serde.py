from __future__ import annotations

import abc
import io
import json
import logging
import pickle
import posixpath
import typing as t
from urllib.parse import unquote
from urllib.parse import urlparse

import attrs
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile
from typing_extensions import get_args

from _bentoml_sdk.typing_utils import is_list_type
from _bentoml_sdk.typing_utils import is_union_type
from _bentoml_sdk.validators import DataframeSchema
from _bentoml_sdk.validators import TensorSchema

if t.TYPE_CHECKING:
    from starlette.requests import Request

    from _bentoml_sdk import IODescriptor


T = t.TypeVar("T", bound="IODescriptor")
logger = logging.getLogger("bentoml.io")


@attrs.frozen
class Payload:
    data: t.Iterable[bytes | memoryview]
    metadata: t.Mapping[str, str] = attrs.field(factory=dict)

    def total_bytes(self) -> int:
        return sum(len(d) for d in self.data)

    @property
    def headers(self) -> t.Mapping[str, str]:
        return {"content-length": str(self.total_bytes()), **self.metadata}


@attrs.frozen
class SerializationInfo:
    mode: str

    def mode_is_json(self) -> bool:
        return self.mode == "json"


class Serde(abc.ABC):
    media_type: str

    @abc.abstractmethod
    def serialize_model(self, model: IODescriptor) -> Payload: ...

    @abc.abstractmethod
    def deserialize_model(self, payload: Payload, cls: type[T]) -> T: ...

    @abc.abstractmethod
    def serialize(self, obj: t.Any, schema: dict[str, t.Any]) -> Payload: ...

    @abc.abstractmethod
    def deserialize(self, payload: Payload, schema: dict[str, t.Any]) -> t.Any: ...

    async def parse_request(self, request: Request, cls: type[T]) -> T:
        """Parse a input model from HTTP request"""
        json_str = await request.body()
        return self.deserialize_model(
            Payload((json_str,), metadata=request.headers), cls
        )


class GenericSerde:
    def _encode(self, obj: t.Any, schema: dict[str, t.Any]) -> t.Any:
        mode = "json" if isinstance(self, JSONSerde) else "python"
        info = SerializationInfo(mode=mode)
        if schema.get("type") == "tensor":
            child_schema = TensorSchema(
                format=schema.get("format", ""),
                dtype=schema.get("dtype"),
                shape=schema.get("shape"),
            )
            return child_schema.encode(child_schema.validate(obj), info)
        if schema.get("type") == "dataframe":
            child_schema = DataframeSchema(
                orient=schema.get("orient", "records"), columns=schema.get("columns")
            )
            return child_schema.encode(child_schema.validate(obj), info)
        if schema.get("type") == "array" and "items" in schema:
            return [self._encode(v, schema["items"]) for v in obj]
        if schema.get("type") == "object" and schema.get("properties"):
            if isinstance(obj, BaseModel):
                return obj.model_dump(mode=mode)
            return {
                k: self._encode(obj[k], child)
                for k, child in schema["properties"].items()
                if k in obj
            }
        return obj

    def _decode(self, obj: t.Any, schema: dict[str, t.Any]) -> t.Any:
        if schema.get("type") == "tensor":
            child_schema = TensorSchema(
                format=schema.get("format", ""),
                dtype=schema.get("dtype"),
                shape=schema.get("shape"),
            )
            return child_schema.validate(obj)
        if schema.get("type") == "dataframe":
            child_schema = DataframeSchema(
                orient=schema.get("orient", "records"), columns=schema.get("columns")
            )
            return child_schema.validate(obj)
        if schema.get("type") == "array" and "items" in schema:
            return [self._decode(v, schema["items"]) for v in obj]
        if (
            schema.get("type") == "object"
            and schema.get("properties")
            and isinstance(obj, t.Mapping)
        ):
            return {
                k: self._decode(obj[k], child)
                for k, child in schema["properties"].items()
                if k in obj
            }
        return obj

    def serialize(self, obj: t.Any, schema: dict[str, t.Any]) -> Payload:
        return self.serialize_value(self._encode(obj, schema))

    def deserialize(self, payload: Payload, schema: dict[str, t.Any]) -> t.Any:
        return self._decode(self.deserialize_value(payload), schema)

    def serialize_value(self, obj: t.Any) -> Payload:
        raise NotImplementedError

    def deserialize_value(self, payload: Payload) -> t.Any:
        raise NotImplementedError


class JSONSerde(GenericSerde, Serde):
    media_type = "application/json"

    def serialize_model(self, model: IODescriptor) -> Payload:
        return Payload(
            (
                model.model_dump_json(
                    exclude=set(getattr(model, "multipart_fields", set()))
                ).encode("utf-8"),
            )
        )

    def deserialize_model(self, payload: Payload, cls: type[T]) -> T:
        return cls.model_validate_json(b"".join(payload.data) or b"{}")

    def serialize_value(self, obj: t.Any) -> Payload:
        return Payload((json.dumps(obj).encode("utf-8"),))

    def deserialize_value(self, payload: Payload) -> t.Any:
        return json.loads(b"".join(payload.data) or b"{}")


class MultipartSerde(JSONSerde):
    media_type = "multipart/form-data"

    async def ensure_file(self, obj: str | UploadFile) -> UploadFile:
        import httpx

        if isinstance(obj, UploadFile):
            return obj
        async with httpx.AsyncClient() as client:
            obj = obj.strip("\"'")  # The url may be JSON encoded
            logger.debug("Request with URL, downloading file from %s", obj)
            resp = await client.get(obj)
            body = io.BytesIO(await resp.aread())
            parsed = urlparse(obj)
            return UploadFile(
                body,
                size=len(body.getvalue()),
                filename=posixpath.basename(unquote(parsed.path)),
                headers=Headers(raw=resp.headers.raw),
            )

    async def parse_request(self, request: Request, cls: type[T]) -> T:
        form = await request.form()
        data: dict[str, t.Any] = {}
        for k in form:
            if k in cls.multipart_fields:
                value = [await self.ensure_file(v) for v in form.getlist(k)]
                field_annotation = cls.model_fields[k].annotation
                if is_union_type(field_annotation):
                    args = get_args(field_annotation)
                    field_annotation = args[0]
                if is_list_type(field_annotation):
                    data[k] = value
                elif len(value) >= 1:
                    data[k] = value[0]
            else:
                assert isinstance(v := form[k], str)
                try:
                    data[k] = json.loads(v)
                except json.JSONDecodeError:
                    data[k] = v
        return cls.model_validate(data)


class PickleSerde(GenericSerde, Serde):
    media_type = "application/vnd.bentoml+pickle"

    def serialize_model(self, model: IODescriptor) -> Payload:
        model_data = model.model_dump()
        return self.serialize_value(model_data)

    def deserialize_model(self, payload: Payload, cls: type[T]) -> T:
        obj = self.deserialize_value(payload)
        if not isinstance(obj, cls):
            obj = cls.model_validate(obj)
        return obj

    def serialize_value(self, obj: t.Any) -> Payload:
        buffers: list[pickle.PickleBuffer] = []
        main_bytes = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
        data: list[bytes | memoryview] = [main_bytes]
        lengths = [len(main_bytes)]
        for buff in buffers:
            data.append(buff.raw())
            lengths.append(len(data[-1]))
            buff.release()
        metadata = {"buffer-lengths": ",".join(map(str, lengths))}
        return Payload(data, metadata)

    def deserialize_value(self, payload: Payload) -> t.Any:
        if "buffer-lengths" not in payload.metadata:
            return pickle.loads(b"".join(payload.data))
        buffer_lengths = list(map(int, payload.metadata["buffer-lengths"].split(",")))
        data_stream = b"".join(payload.data)
        data = memoryview(data_stream)
        start = buffer_lengths[0]
        main_bytes = data[:start]
        buffers: list[pickle.PickleBuffer] = []
        for length in buffer_lengths[1:]:
            buffers.append(pickle.PickleBuffer(data[start : start + length]))
            start += length
        return pickle.loads(main_bytes, buffers=buffers)


ALL_SERDE: t.Mapping[str, type[Serde]] = {
    s.media_type: s for s in [JSONSerde, PickleSerde, MultipartSerde]
}
# Special case for application/x-www-form-urlencoded
ALL_SERDE["application/x-www-form-urlencoded"] = MultipartSerde
