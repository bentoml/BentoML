from __future__ import annotations

import abc
import io
import json
import pickle
import posixpath
import typing as t
from urllib.parse import unquote
from urllib.parse import urlparse

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


class Serde(abc.ABC):
    media_type: str

    @abc.abstractmethod
    def serialize_model(self, model: IODescriptor) -> bytes:
        ...

    @abc.abstractmethod
    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        ...

    @abc.abstractmethod
    def serialize(self, obj: t.Any, schema: dict[str, t.Any]) -> bytes:
        ...

    @abc.abstractmethod
    def deserialize(self, obj_bytes: bytes, schema: dict[str, t.Any]) -> t.Any:
        ...

    async def parse_request(self, request: Request, cls: type[T]) -> T:
        """Parse a input model from HTTP request"""
        json_str = await request.body()
        return self.deserialize_model(json_str, cls)


class GenericSerde:
    def _encode(self, obj: t.Any, schema: dict[str, t.Any]) -> t.Any:
        if schema.get("type") == "tensor":
            child_schema = TensorSchema(
                format=schema.get("format", ""),
                dtype=schema.get("dtype"),
                shape=schema.get("shape"),
            )
            return child_schema.encode(child_schema.validate(obj))
        if schema.get("type") == "dataframe":
            child_schema = DataframeSchema(
                orient=schema.get("orient", "records"), columns=schema.get("columns")
            )
            return child_schema.encode(child_schema.validate(obj))
        if schema.get("type") == "array" and "items" in schema:
            return [self._encode(v, schema["items"]) for v in obj]
        if schema.get("type") == "object" and schema.get("properties"):
            if isinstance(obj, BaseModel):
                return obj.model_dump()
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

    def serialize(self, obj: t.Any, schema: dict[str, t.Any]) -> bytes:
        return self.serialize_value(self._encode(obj, schema))

    def deserialize(self, obj_bytes: bytes, schema: dict[str, t.Any]) -> t.Any:
        return self._decode(self.deserialize_value(obj_bytes), schema)

    def serialize_value(self, obj: t.Any) -> bytes:
        raise NotImplementedError

    def deserialize_value(self, obj_bytes: bytes) -> t.Any:
        raise NotImplementedError


class JSONSerde(GenericSerde, Serde):
    media_type = "application/json"

    def serialize_model(self, model: IODescriptor) -> bytes:
        return model.model_dump_json(
            exclude=set(getattr(model, "multipart_fields", set()))
        ).encode("utf-8")

    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        return cls.model_validate_json(model_bytes)

    def serialize_value(self, obj: t.Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def deserialize_value(self, obj_bytes: bytes) -> t.Any:
        return json.loads(obj_bytes)


class MultipartSerde(JSONSerde):
    media_type = "multipart/form-data"

    async def ensure_file(self, obj: str | UploadFile) -> UploadFile:
        import httpx

        if isinstance(obj, UploadFile):
            return obj
        async with httpx.AsyncClient() as client:
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

    def serialize_model(self, model: IODescriptor) -> bytes:
        model_data = model.model_dump()
        return pickle.dumps(model_data)

    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        obj = pickle.loads(model_bytes)
        if not isinstance(obj, cls):
            obj = cls.model_validate(obj)
        return obj

    def serialize_value(self, obj: t.Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize_value(self, obj_bytes: bytes) -> t.Any:
        return pickle.loads(obj_bytes)


class ArrowSerde(Serde):
    media_type = "application/vnd.bentoml+arrow"

    def serialize_model(self, model: IODescriptor) -> bytes:
        from .arrow import serialize_to_arrow

        buffer = io.BytesIO()
        serialize_to_arrow(model, buffer)
        return buffer.getvalue()

    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        from .arrow import deserialize_from_arrow

        buffer = io.BytesIO(model_bytes)
        return deserialize_from_arrow(cls, buffer)

    def serialize(self, obj: t.Any, schema: dict[str, t.Any]) -> bytes:
        raise NotImplementedError(
            "Serializing arbitrary object to Arrow is not supported"
        )

    def deserialize(self, obj_bytes: bytes, schema: dict[str, t.Any]) -> t.Any:
        raise NotImplementedError(
            "Deserializing arbitrary object from Arrow is not supported"
        )


ALL_SERDE: t.Mapping[str, type[Serde]] = {
    s.media_type: s for s in [JSONSerde, PickleSerde, ArrowSerde, MultipartSerde]
}
# Special case for application/x-www-form-urlencoded
ALL_SERDE["application/x-www-form-urlencoded"] = MultipartSerde
