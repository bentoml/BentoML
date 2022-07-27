import logging
from timeit import default_timer

from opentelemetry import trace

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.server.grpc.interceptors import AsyncServerInterceptor


class AccessLogInterceptor(AsyncServerInterceptor):
    def __init__(self) -> None:
        self.logger = logging.getLogger("bentoml.access")

    async def intercept(
        self,
        method,
        request,
        context,
        method_name: str,
    ) -> None:
        tracer_provider = BentoMLContainer.tracer_provider.get()
        tracer = tracer_provider.get_tracer("opentelemetry.instrumentation.grpc")
        span = tracer.start_span("grpc")

        start = default_timer()
        with trace.use_span(span, end_on_exit=True):
            response = await method(request, context)
        latency = max(default_timer() - start, 0)

        request_info: str = f"api_name={request.api_name},type={str(response.contents.WhichOneof('kind'))[:-6]},size={request.contents.ByteSize()}"
        response_info: str = f"status={context.code()},type={response.contents.WhichOneof('kind')[:-6]},size={response.contents.ByteSize()}"
        trace_and_span: str = f"trace={span.context.trace_id},span={span.context.span_id},sampled={1 if span.get_span_context().trace_flags.sampled else 0}"

        self.logger.info(
            "%s (%s) (%s) %.3fms (%s)",
            context.peer(),
            request_info,
            response_info,
            latency,
            trace_and_span,
        )

        return response
