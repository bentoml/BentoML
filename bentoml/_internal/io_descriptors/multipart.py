from typing import Any, NewType, Tuple

from bentoml.exceptions import InvalidArgument

from ..types import HTTPRequest, HTTPResponse
from .base import IODescriptor

MultipartIO = NewType("MultipartIO", Tuple[Any, ...])


class Multipart(IODescriptor):
    def __init__(self, *items: Tuple[IODescriptor, ...]):
        for i in items:
            if i.name is None:
                # TODO: should the item name be set to index by default?
                raise InvalidArgument("Multipart IO must specify name for each io item")
        self.items = items

    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> MultipartIO:
        pass

    def to_http_response(self, *obj: MultipartIO) -> HTTPResponse:
        pass
