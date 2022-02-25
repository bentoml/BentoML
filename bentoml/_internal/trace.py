import typing as t
from contextvars import ContextVar

from opentelemetry import trace  # type: ignore[import]


class ServiceContextClass:
    def __init__(self) -> None:
        self.request_id_var = ContextVar(
            "request_id_var", default=t.cast("t.Optional[int]", None)
        )
        self.component_name_var: ContextVar[str] = ContextVar(
            "component_name", default="cli"
        )

    @property
    def sampled(self) -> t.Optional[int]:
        span = trace.get_current_span()
        if span is None:
            return None
        return 1 if span.get_span_context().trace_flags.sampled else 0

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

    @property
    def component_name(self) -> t.Optional[str]:
        return self.component_name_var.get()


ServiceContext = ServiceContextClass()
