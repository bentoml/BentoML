import typing as t
from typing import TYPE_CHECKING

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response

from ...exceptions import BentoMLException, InvalidArgument
from ..utils.formparser import (
    concat_to_multipart_responses,
    populate_multipart_requests,
)
from .base import IODescriptor

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa

    from ..types import FileLike  # noqa
    from .image import ImageType  # noqa
    from .json import JSONType  # noqa


_DescriptorType = t.Union[
    str, "JSONType", "FileLike", "ImageType", "np.ndarray[t.Any, np.dtype[t.Any]]"
]

MultipartIO = t.Dict[str, _DescriptorType]


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

    def __init__(self, **inputs: IODescriptor[_DescriptorType]):
        for descriptor in inputs.values():
            if isinstance(descriptor, Multipart):
                raise InvalidArgument(
                    "Multipart IO can not contain nested Multipart item"
                )

        self._inputs = inputs

    def openapi_schema(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        schema: t.Dict[str, t.Dict[str, t.Any]] = {
            "multipart/form-data": {"schema": {"type": "object", "properties": {}}}
        }
        return schema

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return self.openapi_schema()

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return self.openapi_schema()

    async def from_http_request(self, request: Request) -> MultipartIO:
        ctype, _ = parse_options_header(request.headers["content-type"])
        if ctype != b"multipart/form-data":
            raise BentoMLException(
                f"{self.__class__.__name__} only accepts `multipart/form-data` as Content-Type header, got {ctype} instead."
            )

        res = dict()  # type: MultipartIO
        reqs = await populate_multipart_requests(request)

        for k, i in self._inputs.items():
            req = reqs[k]
            v = await i.from_http_request(req)
            res[k] = v
        return res

    async def to_http_response(self, obj: MultipartIO) -> Response:
        res_mapping: t.Dict[str, Response] = {}
        for k, io_ in self._inputs.items():
            data = obj[k]
            res_mapping[k] = await io_.to_http_response(data)
        return await concat_to_multipart_responses(res_mapping)
