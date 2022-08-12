from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import Response

from bentoml.exceptions import BadInput
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
    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import BentoServicerContext

    from ..context import InferenceApiContext as Context
else:
    pb = LazyLoader("pb", globals(), "bentoml.grpc.v1.service_pb2")

MIME_TYPE = "text/plain"


class Text(IODescriptor[str], proto_field="text"):
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

    def __init__(self, *args: t.Any, packed: bool = False, **kwargs: t.Any):
        if args or kwargs:
            raise BentoMLException(
                "'Text' is not designed to take any args or kwargs during initialization."
            )

        self._mime_type = MIME_TYPE
        self._packed = packed

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
                media_type=MIME_TYPE,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return Response(obj, media_type=MIME_TYPE)

    async def from_grpc_request(
        self, request: pb.Request, context: BentoServicerContext
    ) -> str:
        from ..utils.grpc import get_field
        from ..utils.grpc import deserialize_proto
        from ..utils.grpc import raise_grpc_exception

        field = get_field(request, self)

        deserialized = deserialize_proto(request)[field]

        if self._packed and field != "raw_value":
            raise_grpc_exception(
                f"'packed={self._packed}' only accepts 'raw_value'.",
                context=context,
                exc_cls=BadInput,
            )

        if field == "string_value":
            return str(deserialized)

        # { 'content': b'string_content' }
        if "content" not in deserialized:
            raise_grpc_exception(
                "'content' cannot be None.", context=context, exc_cls=BentoMLException
            )

        value = deserialized["content"]
        return value.decode("utf-8")

    async def to_grpc_response(
        self, obj: str, context: BentoServicerContext
    ) -> pb.Response:
        from ..utils.grpc import get_grpc_content_type

        context.set_trailing_metadata(
            (("content-type", get_grpc_content_type(message_format="text")),)
        )

        if self._packed:
            return pb.Response(
                output=pb.Value(raw_value=pb.Raw(content=obj.encode("utf-8")))
            )

        return pb.Response(output=pb.Value(string_value=obj))

    def generate_protobuf(self):
        pass
