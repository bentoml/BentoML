import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import Message

from ...exceptions import BentoMLException, InvalidArgument
from ..utils._formparser import (  # noqa
    concat_to_multipart_responses,
    populate_multipart_requests,
)
from .base import IODescriptor

MultipartIO = t.Dict[str, t.Any]


class Multipart(IODescriptor[MultipartIO]):
    """
    Examples::

    from bentoml.io import Image, JSON, Multipart

    spec = Multipart(img=Image(), annotations=JSON())

    @svc.api(input=spec, output=spec)
    def predict(img, annotations):
        ...
        return img, annotations

    curl -i -F img=@test.jpg -F annotations=@test.json localhost:5000/predict
    """

    def __init__(self, **inputs: IODescriptor):
        for descriptor in inputs.values():
            if not isinstance(descriptor, IODescriptor):
                raise InvalidArgument(
                    "Multipart IO item must be instance of another IODescriptor type"
                )
            if isinstance(descriptor, Multipart):
                raise InvalidArgument(
                    "Multipart IO can not contain nested Multipart item"
                )

        self._inputs = inputs

    @staticmethod
    def _return_body(data: bytes) -> Message:
        return {
            "type": "http.request",
            "body": data,
            "more_body": False,
        }

    def openapi_request_schema(self):
        pass

    def openapi_responses_schema(self):
        pass

    async def from_http_request(self, request: Request) -> MultipartIO:
        ctype, _ = parse_options_header(request.headers["content-type"])
        if ctype != b"multipart/form-data":
            raise BentoMLException(
                f"{self.__class__.__name__} only accepts `multipart/form-data` as Content-Type header, got {ctype} instead."
            )

        res = dict()
        reqs = await populate_multipart_requests(request)

        for k, i in self._inputs.items():
            req = reqs[k]
            v = await i.from_http_request(req)
            res[k] = v
        return res

    async def to_http_response(self, obj: MultipartIO) -> Response:
        res: t.List[t.Tuple[str, t.Union[Response, StreamingResponse]]] = list()
        for i, io_ in enumerate(self._inputs.items()):
            io_descriptor = t.cast(IODescriptor, io_[1])
            r = await io_descriptor.to_http_response(obj[i])  # noqa
            res.append((io_[0], r))
        return await concat_to_multipart_responses(res)
