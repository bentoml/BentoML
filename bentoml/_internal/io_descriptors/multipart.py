import itertools
import json
import typing as t
from functools import partial

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message

from bentoml._internal.utils.http import parse_multipart_to_multireq

from ...exceptions import BentoMLException, InvalidArgument
from .base import IODescriptor

MultipartIO = t.Dict[str, t.Any]


class Multipart(IODescriptor[MultipartIO]):
    """
    Example:

    from bentoml.io import Image, JSON, Multipart
    @svc.api(input=Multipart(image=Image(), annotations=JSON()}), output=JSON())
    def predict(image: "PIL.Image.Image", annotations: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        ...
        return {'img': img.toarray(), 'annotations': annotation}

    curl -i -F image=@test.jpg -F annotations=@test.json localhost:5000/predict
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

        res: t.Dict[str, t.Any] = dict()
        reqs = await parse_multipart_to_multireq(request)

        for k, i in self._inputs.items():
            req = reqs[k]
            v = await i.from_http_request(req)
            res[k] = v
        return res

    async def to_http_response(self, obj: MultipartIO) -> Response:
        """to_http_response.

        :param obj:
        :type obj: MultipartIO
        :rtype: Response
        """
        assert False  # TODO(jiang): not tested yet
        res: t.Dict[str, Response] = dict()
        for k, i in self._inputs.items():
            v = obj.get(k)
            r = await i.to_http_response(v)
            res[k] = r
        return multireq_to_multipart(res)
