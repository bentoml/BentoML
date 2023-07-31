from __future__ import annotations

import typing as t

from starlette.requests import Request
from starlette.responses import ContentStream
from starlette.responses import StreamingResponse

from ...exceptions import BentoMLException
from ..service.openapi import SUCCESS_DESCRIPTION
from ..service.openapi.specification import MediaType
from ..service.openapi.specification import Schema
from ..utils.http import set_cookies
from ..utils.lazy_loader import LazyLoader
from .base import IODescriptor

if t.TYPE_CHECKING:
    from google.protobuf import wrappers_pb2
    from typing_extensions import Self

    from ..context import ServiceContext as Context
    from .base import OpenAPIResponse
else:
    wrappers_pb2 = LazyLoader("wrappers_pb2", globals(), "google.protobuf.wrappers_pb2")

MIME_TYPE = "text/event-stream"


class TextStream(
    IODescriptor[t.AsyncIterable[str]], descriptor_id="bentoml.io.TextStream"
):
    """
    :obj:`TextStream` defines API specification for the inputs/outputs of a Service. :obj:`TextStream`
    represents a stream of string and will be converted from an Asynchronous Generator
    for all outcoming responses as specified in your API function signature.
    Input TextStream is not supported for now.

    Currently, stream based output is only supported with a custom Runnable:

    .. code-block:: python
        :caption: `service.py`

        import bentoml
        import asyncio

        class StreamRunnable(bentoml.Runnable):
            SUPPORTED_RESOURCES = ("cpu",)
            SUPPORTS_CPU_MULTI_THREADING = True

            @bentoml.Runnable.method(stream=True)
            async def count(self, input_text:str):
                for i in range(10):
                    await asyncio.sleep(1)
                    yield f"{input_text} {i}\n"

        stream_runner = bentoml.Runner(SSERunnable)
        svc = bentoml.Service("stream", runners=[stream_runner])

        @svc.api(stream=True, input=bentoml.io.Text(), output=bentoml.io.TextStream())
        async def count(input_text:str):
            ret = sse_runner.count.async_stream(input_text)
            return ret


    Users then can then serve this service with :code:`bentoml serve`:

    .. code-block:: bash

       % bentoml serve ./service.py:svc --reload

    Users can then send requests to the newly started services with any client:

    .. tab-set::

        .. tab-item:: Bash

           .. code-block:: bash

              % curl -X POST http://0.0.0.0:3000/count --d 'Hello World'

    .. note::

        :obj:`TextStream` is not designed to take any ``args`` or ``kwargs`` during initialization.

    Returns:
        :obj:`TextStream`: IO Descriptor that represents stream of string type.
    """

    _mime_type = MIME_TYPE

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        if args or kwargs:
            raise BentoMLException(
                f"'{self.__class__.__name__}' is not designed to take any args or kwargs during initialization."
            ) from None

    def _from_sample(self, sample: str | bytes):
        raise NotImplementedError("TextStream does not support input type for now")

    def input_type(self) -> t.Type[str]:
        raise NotImplementedError("TextStream does not support input type for now")

    def to_spec(self) -> dict[str, t.Any]:
        return {"id": self.descriptor_id}

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> Self:
        return cls()

    def openapi_schema(self) -> Schema:
        return Schema(type="string")

    def openapi_components(self) -> dict[str, t.Any] | None:
        pass

    def openapi_example(self):
        pass

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": SUCCESS_DESCRIPTION,
            "content": {
                self._mime_type: MediaType(
                    schema=self.openapi_schema(), example=self.openapi_example()
                )
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    async def from_http_request(self, request: Request) -> str:
        raise NotImplementedError("TextStream does not support input type for now")

    async def to_http_response(
        self, obj: ContentStream, ctx: Context | None = None
    ) -> StreamingResponse:
        if ctx is not None:
            res = StreamingResponse(
                obj,
                media_type=self._mime_type,
                headers=ctx.response.metadata,  # type: ignore (bad starlette types)
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
            return res
        else:
            return StreamingResponse(obj, media_type=self._mime_type)

    async def from_proto(self, field: wrappers_pb2.StringValue | bytes) -> str:
        raise NotImplementedError("TextStream does not support grpc for now")

    async def to_proto(self, obj: str) -> wrappers_pb2.StringValue:
        raise NotImplementedError("TextStream does not support grpc for now")
