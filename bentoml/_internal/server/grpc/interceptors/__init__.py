from __future__ import annotations

import typing as t
import logging
import functools
from timeit import default_timer
from typing import TYPE_CHECKING

import grpc
from grpc import aio

from ....utils import LazyLoader
from ....utils.grpc import ProtoCodec
from ....utils.grpc import to_http_status
from ....utils.grpc import wrap_rpc_handler
from ....utils.grpc import get_grpc_content_type
from ....utils.grpc.codec import GRPC_CONTENT_TYPE

if TYPE_CHECKING:
    from grpc.aio._typing import MetadataType

    from bentoml.grpc.v1 import service_pb2

    from ..types import Request
    from ..types import Response
    from ..types import RpcMethodHandler
    from ..types import AsyncHandlerMethod
    from ..types import HandlerCallDetails
    from ..types import BentoServicerContext
    from ....utils.grpc.codec import Codec
else:
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")

logger = logging.getLogger(__name__)


class GenericHeadersServerInterceptor(aio.ServerInterceptor):
    """
    A light header interceptor that provides some initial metadata to the client.
    TODO: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/HEAD/doc/PROTOCOL-HTTP2.md
    """

    def __init__(self, *, codec: Codec | None = None):
        if not codec:
            # By default, we use ProtoCodec.
            codec = ProtoCodec()
        self._codec = codec

    def set_trailing_metadata(self, context: BentoServicerContext):
        # We want to send some initial metadata to the client.
        # gRPC doesn't use `:status` pseudo header to indicate success or failure
        # of the current request. gRPC instead uses trailers for this purpose, and
        # trailers are sent during `send_trailing_metadata` call
        # For now we are sending over the content-type header.
        headers = [("content-type", get_grpc_content_type(codec=self._codec))]
        context.set_trailing_metadata(headers)

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
                resp = behaviour(request, context)
                if not hasattr(resp, "__aiter__"):
                    resp = await resp
                return resp

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)


class AccessLogServerInterceptor(aio.ServerInterceptor):
    """
    An asyncio interceptors for access log.
    """

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        logger = logging.getLogger("bentoml.access")
        handler = await continuation(handler_call_details)
        method_name = handler_call_details.method

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:

                content_type = GRPC_CONTENT_TYPE

                trailing_metadata: MetadataType | None = context.trailing_metadata()
                if trailing_metadata:
                    trailing = dict(trailing_metadata)
                    content_type = trailing.get("content-type", GRPC_CONTENT_TYPE)

                start = default_timer()
                try:
                    response = behaviour(request, context)
                    if not hasattr(response, "__aiter__"):
                        response = await response
                except Exception as e:
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(str(e))
                    response = service_pb2.Response()
                finally:
                    latency = max(default_timer() - start, 0)
                    req = [
                        "scheme=http",  # TODO: support https when ssl is added
                        f"path={method_name}",
                        f"type={content_type}",
                        f"size={request.ByteSize()}",
                    ]
                    resp = [
                        f"http_status={to_http_status(context.code())}",
                        f"grpc_status={context.code().value[0]}",
                        f"type={content_type}",
                        f"size={response.ByteSize()}",
                    ]

                    # TODO: fix ports
                    logger.info(
                        f"{context.peer()} ({','.join(req)}) ({','.join(resp)}) {latency:.3f}ms"
                    )
                return response

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
