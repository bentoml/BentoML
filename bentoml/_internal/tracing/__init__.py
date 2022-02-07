import contextvars
import typing as t

from opentelemetry import trace  # type: ignore[import]


class ServiceContextClass:
    def __init__(self) -> None:
        self.request_id_var = contextvars.ContextVar(
            "request_id_var", default=t.cast("t.Optional[int]", None)
        )

    @property
    def trace_id(self) -> t.Optional[int]:
        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().trace_id

    @property
    def span_id(self) -> t.Optional[int]:
        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().span_id

    @property
    def request_id(self) -> t.Optional[int]:
        return self.request_id_var.get()


ServiceContext = ServiceContextClass()
