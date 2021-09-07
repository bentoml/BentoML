from typing import Any, Dict, NewType, Tuple

from bentoml.exceptions import InvalidArgument

from ..types import HTTPRequest, HTTPResponse
from .base import IODescriptor

MultipartIO = NewType("MultipartIO", Tuple[Any, ...])


class Multipart(IODescriptor):
    """
    Example:

    from bentoml.io import Image, JSON, Multipart
    @svc.api(input=Multipart(img=Image(), annotation=JSON()}), output=JSON())
    """

    def __init__(self, **inputs: Dict[str, IODescriptor]):
        for name, descriptor in inputs.items():
            if not isinstance(descriptor, IODescriptor):
                raise InvalidArgument(
                    "Multipart IO item must be instance of another IODescriptor type"
                )
            if isinstance(descriptor, Multipart):
                raise InvalidArgument(
                    "Multipart IO can not contain nested Multipart item"
                )

        self._inputs = inputs

    def openapi_schema(self):
        pass

    def from_http_request(self, request: HTTPRequest) -> MultipartIO:
        pass

    def to_http_response(self, *obj: MultipartIO) -> HTTPResponse:
        pass
