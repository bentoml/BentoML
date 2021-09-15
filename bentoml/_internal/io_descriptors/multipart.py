import typing as t

from starlette.requests import Request
from starlette.responses import Response

from bentoml.exceptions import InvalidArgument

from .base import IODescriptor

MultipartIO = t.NewType("MultipartIO", t.Tuple[t.Any, ...])


class Multipart(IODescriptor):
    """
    Example:

    from bentoml.io import Image, JSON, Multipart
    @svc.api(input=Multipart(img=Image(), annotation=JSON()}), output=JSON())
    """

    def __init__(self, **inputs: t.Dict[str, IODescriptor]):
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

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    async def from_http_request(self, request: Request) -> MultipartIO:
        pass

    async def to_http_response(self, *obj: MultipartIO) -> Response:
        pass
