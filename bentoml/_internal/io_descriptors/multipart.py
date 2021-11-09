import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ...exceptions import BentoMLException, InvalidArgument
from ..utils.formparser import (
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
    output = Multipart(img_output=Image(), annotations_output=JSON())

    @svc.api(input=spec, output=output)
    def predict(img, annotations):
        ...
        return dict(img_output=img, annotations_output=annotations)

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

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""

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

    async def to_http_response(self, obj: MultipartIO) -> StreamingResponse:
        res: t.List[t.Tuple[str, Response]] = list()
        for io_, (output_name, output_data) in zip(self._inputs.values(), obj.items()):
            r: Response = await io_.to_http_response(output_data)
            res.append((output_name, r))
        return await concat_to_multipart_responses(res)
