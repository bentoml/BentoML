from __future__ import annotations

import typing as t
import functools
from typing import TYPE_CHECKING

from grpc import aio

from bentoml.grpc.utils import wrap_rpc_handler
from bentoml.grpc.utils import GRPC_CONTENT_TYPE

if TYPE_CHECKING:

    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext


class GenericHeadersServerInterceptor(aio.ServerInterceptor):
    """
    A light header interceptor that provides some initial metadata to the client.
    Refers to https://chromium.googlesource.com/external/github.com/grpc/grpc/+/HEAD/doc/PROTOCOL-HTTP2.md
    """

    def __init__(self, *, message_format: str | None = None):
        if not message_format:
            # By default, we are sending proto message.
            message_format = "proto"
        self._content_type = f"{GRPC_CONTENT_TYPE}+{message_format}"

    def set_trailing_metadata(self, context: BentoServicerContext):
        # We want to send some initial metadata to the client.
        # gRPC doesn't use `:status` pseudo header to indicate success or failure
        # of the current request. gRPC instead uses trailers for this purpose, and
        # trailers are sent during `send_trailing_metadata` call
        # For now we are sending over the content-type header.
        context.set_trailing_metadata((("content-type", self._content_type),))

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                # setup metadata
                self.set_trailing_metadata(context)

                # for the rpc itself.
                return await behaviour(request, context)

            return new_behaviour

        return t.cast("RpcMethodHandler", wrap_rpc_handler(wrapper, handler))
