from ..types import HTTPRequest, HTTPResponse
from .base import IODescriptor


class Str(IODescriptor):
    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> str:
        pass

    def to_http_response(self, obj: str) -> HTTPResponse:
        pass
