import typing as t
import contextvars


class ServiceTraceContext:
    def __init__(self) -> None:
        self._request_id_var = contextvars.ContextVar(
            "_request_id_var", default=t.cast("t.Optional[int]", None)
        )

    @property
    def trace_id(self) -> t.Optional[int]:
        from opentelemetry import trace  # type: ignore

        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().trace_id

    @property
    def span_id(self) -> t.Optional[int]:
        from opentelemetry import trace  # type: ignore

        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().span_id

    @property
    def request_id(self) -> t.Optional[int]:
        """
        Different from span_id, request_id is unique for each inbound request.
        """
        return self._request_id_var.get()

    @request_id.setter
    def request_id(self, request_id: t.Optional[int]) -> None:
        self._request_id_var.set(request_id)

    @request_id.deleter
    def request_id(self) -> None:
        self._request_id_var.set(None)


trace_context = ServiceTraceContext()
