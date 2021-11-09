import typing as t

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import BentoMLException, InvalidArgument
from ..utils.formparser import (  # noqa
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

    async def to_http_response(self, obj: MultipartIO) -> Response:
        res: t.List[t.Tuple[str, Response]] = list()
        for i, io_ in enumerate(
            self._inputs.items()
        ):  # type: t.Tuple[int, t.Tuple[str, IODescriptor]]
            r: Response = await io_[1].to_http_response(obj[i])  # type: ignore[index]
            res.append((io_[0], r))
        return await concat_to_multipart_responses(res)
