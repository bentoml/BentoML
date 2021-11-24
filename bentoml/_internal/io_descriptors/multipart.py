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
    import pandas as pd  # noqa

    from ..types import FileLike
    from .file import File  # noqa
    from .image import Image, ImageType  # noqa
    from .json import JSON, JSONType  # noqa
    from .numpy import NumpyNdarray  # noqa
    from .pandas import PandasDataFrame, PandasSeries  # noqa
    from .text import Text  # noqa

_DescriptorType = t.Union[
    "Image", "JSON", "Text", "NumpyNdarray", "PandasDataFrame", "PandasSeries", "File"
]

MultipartIO = t.Dict[
    str,
    t.Union[
        str,
        "JSONType",
        "FileLike",
        "ImageType",
        "np.ndarray[t.Any, np.dtype[t.Any]]",
        "pd.DataFrame",
        "pd.Series",
    ],
]


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

    def __init__(self, **inputs: _DescriptorType):
        for descriptor in inputs.values():
            if isinstance(descriptor, Multipart):  # pragma: no cover
                raise InvalidArgument(
                    "Multipart IO can not contain nested Multipart item"
                )

        self._inputs = inputs  # type: t.Dict[str, _DescriptorType]

    def openapi_schema_type(self) -> t.Dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                k: io.openapi_schema_type() for k, io in self._inputs.items()
            },
        }

    def openapi_request_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for incoming requests"""
        return {"multipart/form-data": {"schema": self.openapi_schema_type()}}

    def openapi_responses_schema(self) -> t.Dict[str, t.Any]:
        """Returns OpenAPI schema for outcoming responses"""
        return {"multipart/form-data": {"schema": self.openapi_schema_type()}}

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

    @t.overload
    async def to_http_response(self, obj: t.Dict[str, str]) -> Response:
        ...

    @t.overload
    async def to_http_response(self, obj: t.Dict[str, "JSONType"]) -> Response:
        ...

    @t.overload
    async def to_http_response(self, obj: t.Dict[str, "ImageType"]) -> Response:
        ...

    @t.overload
    async def to_http_response(
        self, obj: t.Dict[str, "np.ndarray[t.Any, np.dtype[t.Any]]"]
    ) -> Response:
        ...

    @t.overload
    async def to_http_response(self, obj: t.Dict[str, "FileLike"]) -> Response:
        ...

    @t.overload
    async def to_http_response(
        self, obj: t.Dict[str, t.Union["pd.DataFrame", "pd.Series"]]
    ) -> Response:
        ...

    async def to_http_response(self, obj: MultipartIO) -> Response:
        res_mapping: t.Dict[str, Response] = {}
        for k, io_ in self._inputs.items():
            data = obj[k]
            res_mapping[k] = await io_.to_http_response(data)
        return await concat_to_multipart_responses(res_mapping)
