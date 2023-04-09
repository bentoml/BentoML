from __future__ import annotations

import json
import logging
import typing as t

from pydantic import create_model_from_typeddict
from starlette.requests import Request
from starlette.responses import Response

from .base import IODescriptor
from .json import DefaultJsonEncoder, parse_dict_to_proto
from ..service.openapi import REF_PREFIX
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import Schema
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.http import set_cookies
from ...exceptions import BadInput
from ...exceptions import InvalidArgument

EXC_MSG = "'pydantic' must be installed to use 'pydantic_model'. Install with 'pip install bentoml[io-json]'."

if t.TYPE_CHECKING:
    from types import UnionType

    import pydantic.schema as schema
    from google.protobuf import struct_pb2
    from typing_extensions import Self

    from .base import OpenAPIResponse
    from ..context import InferenceApiContext as Context

else:
    pydantic = LazyLoader("pydantic", globals(), "pydantic", exc_msg=EXC_MSG)
    schema = LazyLoader("schema", globals(), "pydantic.schema", exc_msg=EXC_MSG)
    # lazy load our proto generated.
    struct_pb2 = LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")
    # lazy load numpy for processing ndarray.
    np = LazyLoader("np", globals(), "numpy")


TypedDictType = t.Union[t.Dict[str, t.Any], t.TypedDict, None]

logger = logging.getLogger(__name__)


class TypedJSON(IODescriptor[TypedDictType], descriptor_id="bentoml.io.TypedJSON", proto_fields=("json",)):
    # default mime type is application/json
    _mime_type = "application/json"

    def __init__(
            self,
            *,
            typed_dict: t.Type[t.TypedDict] | t.Type[t.List[t.TypedDict]],
            sample: TypedDictType | None= None,
            json_encoder: t.Type[json.JSONEncoder] = DefaultJsonEncoder,
    ):

        if typed_dict is not dict:  # there is no way to check isinstance TypedDict in runtime
            assert issubclass(
                typed_dict, dict
            ), "'typed_dict' must be a subclass of 'typing.TypedDict or dict'."
        self._pydantic_model = create_model_from_typeddict(typed_dict)
        self.sample = sample
        self._json_encoder = json_encoder

    def _from_sample(self, sample: TypedDictType) -> TypedDictType:
        raise BadInput("TypedJSON Does not support TypedJSON.from_sample(), "
                       "use TypedJson(typed_dict=HelloTDict, sample={\"hello\": \"world\"}) "
                       "instead of TypedJSON.from_sample({\"hello\": \"world\"} )")

    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": self.descriptor_id,
            "args": {
                "has_pydantic_model": self._pydantic_model is not None,
                "has_json_encoder": self._json_encoder is not None,
            },
        }

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> Self:
        if "args" not in spec:
            raise InvalidArgument(f"Missing args key in JSON spec: {spec}")
        if "has_pydantic_model" in spec["args"] and spec["args"]["has_pydantic_model"]:
            logger.warning(
                "BentoML does not support loading pydantic models from URLs; output will be a normal dictionary."
            )
        if "has_json_encoder" in spec["args"] and spec["args"]["has_json_encoder"]:
            logger.warning(
                "BentoML does not support loading JSON encoders from URLs; output will be a normal dictionary."
            )

        return cls()

    def input_type(self) -> UnionType:
        return TypedDictType

    def openapi_schema(self) -> Schema:
        if not self._pydantic_model:
            return Schema(type="object")

        # returns schemas from pydantic_model.
        return Schema(
            **schema.model_process_schema(
                self._pydantic_model,
                model_name_map=schema.get_model_name_map(
                    schema.get_flat_models_from_model(self._pydantic_model)
                ),
                ref_prefix=REF_PREFIX,
            )[0]
        )

    def openapi_components(self) -> dict[str, t.Any] | None:
        if not self._pydantic_model:
            return {}

        from ..service.openapi.utils import pydantic_components_schema

        return {"schemas": pydantic_components_schema(self._pydantic_model)}

    def openapi_example(self):
        if self.sample is not None:
            if LazyType["pydantic.BaseModel"]("pydantic.BaseModel").isinstance(
                self.sample
            ):
                return self.sample.dict()
            elif isinstance(self.sample, (str, list)):
                return json.dumps(
                    self.sample,
                    cls=self._json_encoder,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=None,
                    separators=(",", ":"),
                )
            elif isinstance(self.sample, dict):
                return self.sample

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    async def from_http_request(self, request: Request) -> TypedDictType:
        json_str = await request.body()
        try:
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise BadInput(f"Invalid JSON input received: {e}") from None

        return json_obj

    async def to_http_response(
        self, obj: TypedDictType, ctx: Context | None = None
    ):
        json_str = (
            json.dumps(
                obj,
                cls=self._json_encoder,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            )
            if obj is not None
            else None
        )

        if ctx is not None:
            res = Response(
                json_str,
                media_type=self._mime_type,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(json_str, media_type=self._mime_type)

    async def from_proto(self, field: struct_pb2.Value | bytes) -> TypedDictType:
        from google.protobuf.json_format import MessageToDict

        if isinstance(field, bytes):
            content = field
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                raise BadInput(f"Invalid JSON input received: {e}") from None
        else:
            assert isinstance(field, struct_pb2.Value)
            parsed = MessageToDict(field, preserving_proto_field_name=True)

        return parsed

    async def to_proto(self, obj: TypedDictType) -> struct_pb2.Value:
        msg = struct_pb2.Value()
        return parse_dict_to_proto(obj, msg, json_encoder=self._json_encoder)

