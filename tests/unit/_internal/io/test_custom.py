from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

import bentoml
from bentoml.io import IODescriptor
from bentoml.exceptions import BentoMLException
from bentoml._internal.utils.http import set_cookies
from bentoml._internal.service.openapi import SUCCESS_DESCRIPTION
from bentoml._internal.service.openapi.specification import Schema
from bentoml._internal.service.openapi.specification import MediaType

if TYPE_CHECKING:
    from google.protobuf import wrappers_pb2

    from bentoml._internal.context import ServiceContext as Context
    from bentoml._internal.io_descriptors.base import OpenAPIResponse


# testing the minimal required IO descriptor to ensure we don't break
# compatibility with custom descriptors when implementing new features
# in IODescriptor.
class CustomDescriptor(IODescriptor[str]):
    _mime_type = "text/custom"

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        if args or kwargs:
            raise BentoMLException(
                f"'{self.__class__.__name__}' is not designed to take any args or kwargs during initialization."
            ) from None

    def input_type(self) -> t.Type[str]:
        return str

    def _from_sample(self, sample: str | bytes) -> str:
        if isinstance(sample, bytes):
            sample = sample.decode("utf-8")
        return sample

    def openapi_schema(self) -> Schema:
        return Schema(type="string")

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self):
        return str(self.sample)

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
        }

    async def from_http_request(self, request: Request) -> str:
        body = await request.body()
        return body.decode("cp1252")

    async def to_http_response(self, obj: str, ctx: Context | None = None) -> Response:
        if ctx is not None:
            res = Response(
                obj,
                media_type=self._mime_type,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(obj, media_type=self._mime_type)

    async def from_proto(self, field: wrappers_pb2.StringValue | bytes) -> str:
        if isinstance(field, bytes):
            return field.decode("cp1252")
        else:
            assert isinstance(field, wrappers_pb2.StringValue)
            return field.value

    async def to_proto(self, obj: str) -> wrappers_pb2.StringValue:
        return wrappers_pb2.StringValue(value=obj)


def test_custom_io_descriptor():
    svc = bentoml.Service("test")

    @svc.api(input=CustomDescriptor(), output=CustomDescriptor())
    def descriptor_test_api(inp):
        return inp

    svc.asgi_app
