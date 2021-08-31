from typing import Union

from ..types import FileLike, HTTPRequest, HTTPResponse
from .base import IODescriptor


class File(IODescriptor):
    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> Union[bytes, FileLike]:
        pass

    def to_http_response(self, obj: Union[bytes, FileLike]) -> HTTPResponse:
        pass
