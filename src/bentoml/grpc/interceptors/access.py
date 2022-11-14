from __future__ import annotations

import typing as t
import logging
import functools
from timeit import default_timer
from typing import TYPE_CHECKING

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import to_http_status
from bentoml.grpc.utils import wrap_rpc_handler
from bentoml.grpc.utils import GRPC_CONTENT_TYPE
from bentoml.grpc.utils import import_generated_stubs

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc.aio._typing import MetadataType  # pylint: disable=unused-import

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext
else:
    pb, _ = import_generated_stubs()
    grpc, aio = import_grpc()


class AccessLogServerInterceptor(aio.ServerInterceptor):
    """
    An asyncio interceptor for access logging.
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

                response = pb.Response()
                start = default_timer()
                try:
                    response = await behaviour(request, context)
                except Exception as e:  # pylint: disable=broad-except
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(str(e))
                finally:
                    latency = max(default_timer() - start, 0) * 1000

                    req = [
                        "scheme=http",  # TODO: support https when ssl is added
                        f"path={method_name}",
                        f"type={content_type}",
                        f"size={request.ByteSize()}",
                    ]

                    # Note that in order AccessLogServerInterceptor to work, the
                    # interceptor must be added to the server after AsyncOpenTeleServerInterceptor
                    # and PrometheusServerInterceptor.
                    typed_context_code = t.cast(grpc.StatusCode, context.code())
                    resp = [
                        f"http_status={to_http_status(typed_context_code)}",
                        f"grpc_status={typed_context_code.value[0]}",
                        f"type={content_type}",
                        f"size={response.ByteSize()}",
                    ]

                    logger.info(
                        "%s (%s) (%s) %.3fms",
                        context.peer(),
                        ",".join(req),
                        ",".join(resp),
                        latency,
                    )
                return response

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
