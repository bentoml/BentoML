from __future__ import annotations

import typing as t
import logging
import functools
from timeit import default_timer
from typing import TYPE_CHECKING

from grpc import aio
from opentelemetry import trace

from ....utils.grpc import wrap_rpc_handler

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace.span import Span

    from ..types import Request
    from ..types import Response
    from ..types import HandlerMethod
    from ..types import RpcMethodHandler
    from ..types import HandlerCallDetails
    from ..types import BentoServicerContext

logger = logging.getLogger(__name__)


class AccessLogInterceptor(aio.ServerInterceptor):
    """
    An asyncio interceptors for access log.

    .. TODO:
        - Add support for streaming RPCs.
    """

    def __init__(self, tracer_provider: TracerProvider) -> None:
        self.logger = logging.getLogger("bentoml.access")
        self.tracer_provider = tracer_provider

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)
        method_name = handler_call_details.method

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(
            behaviour: HandlerMethod[Response | t.AsyncGenerator[Response, None]]
        ) -> t.Callable[..., t.Any]:
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response:

                tracer = self.tracer_provider.get_tracer(
                    "opentelemetry.instrumentation.grpc"
                )
                span: Span = tracer.start_span("grpc")
                span_context = span.get_span_context()
                kind = str(request.input.WhichOneof("kind"))

                start = default_timer()
                with trace.use_span(span, end_on_exit=True):
                    response = behaviour(request, context)
                    if not hasattr(response, "__aiter__"):
                        response = await response
                latency = max(default_timer() - start, 0)

                req_info = f"api_name={request.api_name},type={kind},size={request.input.ByteSize()}"
                resp_info = f"status={context.code()},type={kind},size={response.output.ByteSize()}"
                trace_and_span = f"trace={span_context.trace_id},span={span_context.span_id},sampled={1 if span_context.trace_flags.sampled else 0}"

                self.logger.info(
                    f"{context.peer()} ({req_info}) ({resp_info}) {latency:.3f}ms ({trace_and_span})"
                )

                return response

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
