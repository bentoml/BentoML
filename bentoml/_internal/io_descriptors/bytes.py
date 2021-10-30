import typing as t

from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import BentoMLException, InvalidArgument
from ..types import FileLike
from .base import IODescriptor


class Bytes(IODescriptor):
    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

    async def from_http_request(self, request: Request) -> t.Union[bytes, FileLike]:
        pass

    async def to_http_response(self, obj: t.Union[bytes, FileLike]) -> Response:
        pass
