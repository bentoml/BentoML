from typing import Any, NewType, Tuple

from ..types import HTTPRequest, HTTPResponse
from .base import IODescriptor

MultipartIO = NewType("MultipartIO", Tuple[Any, ...])


class Multipart(IODescriptor):
    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> MultipartIO:
        pass

    def to_http_response(self, *obj: MultipartIO) -> HTTPResponse:
        pass
