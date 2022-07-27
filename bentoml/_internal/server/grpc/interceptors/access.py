from __future__ import annotations

import typing as t
import logging
from timeit import default_timer
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide
from opentelemetry import trace

from . import AsyncServerInterceptor
from ....configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace.span import Span

    from bentoml.grpc.v1.service_pb2 import Request
    from bentoml.grpc.v1.service_pb2 import Response

    from ..types import HandlerMethod
    from ..types import BentoServicerContext


class AccessLogInterceptor(AsyncServerInterceptor):
    def __init__(self) -> None:
        self.logger = logging.getLogger("bentoml.access")

    @inject
    async def intercept(
        self,
        method: HandlerMethod[t.Any],
        request: Request,
        context: BentoServicerContext,
        method_name: str,
        *,
        tracer_provider: TracerProvider = Provide[BentoMLContainer.tracer_provider],
    ) -> t.AsyncGenerator[Response, None]:
        tracer = tracer_provider.get_tracer("opentelemetry.instrumentation.grpc")
        span: Span = tracer.start_span("grpc")
        span_context = span.get_span_context()
        kind = str(request.contents.WhichOneof("kind"))

        start = default_timer()
        with trace.use_span(span, end_on_exit=True):
            response_or_iterator = method(request, context)
            if not hasattr(response_or_iterator, "__aiter__"):
                response_or_iterator = await response_or_iterator
        latency = max(default_timer() - start, 0)

        req_info = f"api_name={request.api_name},type={kind},size={request.contents.ByteSize()}"
        resp_info = f"status={context.code()},type={kind},size={response_or_iterator.contents.ByteSize()}"
        trace_and_span = f"trace={span_context.trace_id},span={span_context.span_id},sampled={1 if span_context.trace_flags.sampled else 0}"

        self.logger.info(
            f"{context.peer()} ({req_info}) ({resp_info}) {latency:.3f}ms ({trace_and_span})"
        )

        return response_or_iterator
