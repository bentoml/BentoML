from __future__ import annotations

import logging

import typing as t
from typing import TYPE_CHECKING
from timeit import default_timer

from opentelemetry import trace
from simple_di import Provide
from simple_di import inject

from bentoml.exceptions import BadInput
from ....configuration.containers import BentoMLContainer
from . import AsyncServerInterceptor

if TYPE_CHECKING:
    from opentelemetry.trace.span import Span
    from opentelemetry.sdk.trace import TracerProvider
    from ..types import BentoServicerContext
    from ..types import HandlerMethod
    from bentoml.grpc.v1.service_pb2 import Request
    from bentoml.grpc.v1.service_pb2 import Response


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
        tracer_provider: TracerProvider = Provide[BentoMLContainer.tracer_provider]
    ) -> Response:
        tracer_provider = BentoMLContainer.tracer_provider.get()
        tracer = tracer_provider.get_tracer("opentelemetry.instrumentation.grpc")
        span: Span = tracer.start_span("grpc")
        span_context = span.get_span_context()

        start = default_timer()
        with trace.use_span(span, end_on_exit=True):
            response = method(request, context)
        latency = max(default_timer() - start, 0)

        request_info: str = f"api_name={request.api_name},type={str(response.contents.WhichOneof('kind'))[:-6]},size={request.contents.ByteSize()}"
        response_info = f"status={context.code()},type={kind.strip('_value')},size={response.contents.ByteSize()}"
        trace_and_span = f"trace={span_context.trace_id},span={span_context.span_id},sampled={1 if span_context.trace_flags.sampled else 0}"

        self.logger.info(
            "%s (%s) (%s) %.3fms (%s)",
            context.peer(),
            request_info,
            response_info,
            latency,
            trace_and_span,
        )

        return response
