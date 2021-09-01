import typing as t

from ..types import HTTPRequest, HTTPResponse
from .base import IODescriptor

if t.TYPE_CHECKING:
    import pydantic


class JSON(IODescriptor):
    def __init__(self, pydantic_model: "pydantic.main.BaseModel" = None):
        if pydantic_model is not None:
            self._pydantic_model = pydantic_model

    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> str:
        pass

    def to_http_response(self, obj: str) -> HTTPResponse:
        pass
