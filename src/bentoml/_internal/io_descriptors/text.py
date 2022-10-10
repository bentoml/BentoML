from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from bentoml.exceptions import BentoMLException

from .base import IODescriptor
from ..utils.http import set_cookies
from ..service.openapi import SUCCESS_DESCRIPTION
from ..utils.lazy_loader import LazyLoader
from ..service.openapi.specification import Schema
from ..service.openapi.specification import Response as OpenAPIResponse
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import RequestBody

if TYPE_CHECKING:
    from google.protobuf import wrappers_pb2

    from ..context import InferenceApiContext as Context
else:
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")

MIME_TYPE = "text/plain"


class Text(IODescriptor[str]):
    """
    :obj:`Text` defines API specification for the inputs/outputs of a Service. :obj:`Text`
    represents strings for all incoming requests/outcoming responses as specified in
    your API function signature.

    A sample GPT2 service implementation:

    .. code-block:: python
       :caption: `service.py`

       from __future__ import annotations

       import bentoml
       from bentoml.io import Text

       runner = bentoml.tensorflow.get('gpt2:latest').to_runner()

       svc = bentoml.Service("gpt2-generation", runners=[runner])

       @svc.api(input=Text(), output=Text())
       def predict(text: str) -> str:
           res = runner.run(text)
           return res['generated_text']

    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

       % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

           .. code-block:: bash

              % curl -X POST -H "Content-Type: text/plain" \\
                      --data 'Not for nothing did Orin say that people outdoors.' \\
                      http://0.0.0.0:3000/predict

        .. tab-item:: Python

           .. code-block:: python
              :caption: `request.py`

              import requests
              requests.post(
                  "http://0.0.0.0:3000/predict",
                  headers = {"content-type":"text/plain"},
                  data = 'Not for nothing did Orin say that people outdoors.'
              ).text

    .. note::

        :obj:`Text` is not designed to take any ``args`` or ``kwargs`` during initialization.

    Returns:
        :obj:`Text`: IO Descriptor that represents strings type.
    """

    _proto_fields = ("text",)
    _mime_type = MIME_TYPE

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        if args or kwargs:
            raise BentoMLException(
                f"'{self.__class__.__name__}' is not designed to take any args or kwargs during initialization."
            ) from None

    def input_type(self) -> t.Type[str]:
        return str

    def openapi_schema(self) -> Schema:
        return Schema(type="string")

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_request_body(self) -> RequestBody:
        return RequestBody(
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
            required=True,
        )

    def openapi_responses(self) -> OpenAPIResponse:
        return OpenAPIResponse(
            description=SUCCESS_DESCRIPTION,
            content={self._mime_type: MediaType(schema=self.openapi_schema())},
        )

    async def from_http_request(self, request: Request) -> str:
        obj = await request.body()
        return str(obj.decode("utf-8"))

    async def to_http_response(self, obj: str, ctx: Context | None = None) -> Response:
        if ctx is not None:
            res = Response(
                obj,
                media_type=self._mime_type,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(obj, media_type=self._mime_type)

    async def from_proto(self, field: wrappers_pb2.StringValue | bytes) -> str:
        if isinstance(field, bytes):
            return field.decode("utf-8")
        else:
            assert isinstance(field, wrappers_pb2.StringValue)
            return field.value

    async def to_proto(self, obj: str) -> wrappers_pb2.StringValue:
        return wrappers_pb2.StringValue(value=obj)
