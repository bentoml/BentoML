import itertools
import json
import typing as t
from functools import partial

from multipart.multipart import parse_options_header
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message

from ...exceptions import BentoMLException, InvalidArgument
from .base import IODescriptor

MultipartIO = t.NewType("MultipartIO", t.Dict[str, t.Any])


class Multipart(IODescriptor):
    """
    Example:

    from bentoml.io import Image, JSON, Multipart
    @svc.api(input=Multipart(image=Image(), annotations=JSON()}), output=JSON())
    def predict(image: "PIL.Image.Image", annotations: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        ...
        return {'img': img.toarray(), 'annotations': annotation}

    curl -i -F image=@test.jpg -F annotations=@test.json localhost:5000/predict
    """

    def __init__(self, **inputs: t.Dict[str, IODescriptor]):
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
        reqs, res = {}, []
        headers = request.headers

        ctype, _ = parse_options_header(
            request.headers["content-type"]
        )  # type: t.Tuple[bytes, t.Dict[bytes, bytes]]
        if ctype != b"multipart/form-data":
            raise BentoMLException(
                f"{self.__class__.__name__} only accepts `multipart/form-data` as Content-Type header, got {ctype} instead."
            )

        form_datas = await request.form()
        for params, obj in form_datas.items():
            if params not in self._inputs:
                raise BentoMLException(
                    f"{params} is not defined under users API function."
                )
            data = await obj.read()
            reqs[params] = partial(self._return_body, data=data)
        for i, j in itertools.zip_longest(self._inputs.values(), reqs.values()):
            req = Request(
                scope={
                    "type": "http",
                    "scheme": "http",
                    "method": "POST",
                    "headers": headers,
                },
                receive=j,
            )
            v = await i.from_http_request(req)
            res.append(v)
        return {i: res[j] for j in len(res) for i in reqs}

    async def to_http_response(self, *obj: MultipartIO) -> Response:
        """to_http_response.

        :param obj:
        :type obj: MultipartIO
        :rtype: Response
        """
        pass
