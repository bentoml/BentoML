from __future__ import annotations

import typing as t
import logging
import functools
from timeit import default_timer
from typing import TYPE_CHECKING

import grpc
from grpc import aio

from ....utils import LazyLoader
from ....utils.grpc import to_http_status
from ....utils.grpc import wrap_rpc_handler

if TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2

    from ..types import Request
    from ..types import Response
    from ..types import HandlerMethod
    from ..types import RpcMethodHandler
    from ..types import AsyncHandlerMethod
    from ..types import HandlerCallDetails
    from ..types import BentoServicerContext
else:
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")

logger = logging.getLogger(__name__)

# content-type is always application/grpc
GRPC_CONTENT_TYPE = "application/grpc"


class AccessLogInterceptor(aio.ServerInterceptor):
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
        print(handler)
        method_name = handler_call_details.method

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: HandlerMethod[Response] | AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:

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
                        "scheme=http",
                        "method=POST",  # method is always a POST for gRPC
                        f"path={method_name}",
                        f"api_name={request.api_name}",
                        f"type={GRPC_CONTENT_TYPE}",
                        f"size={request.ByteSize()}",
                    ]
                    resp = [
                        f"http_status={to_http_status(context.code())}",
                        f"status={context.code()}",
                        f"type={GRPC_CONTENT_TYPE}",
                        f"size={response.ByteSize()}",
                    ]

                    logger.info(
                        f"{context.peer()} ({','.join(req)}) ({','.join(resp)}) {latency:.3f}ms"
                    )
                return response

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
