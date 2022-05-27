from __future__ import annotations

from contextvars import ContextVar

from opentelemetry import trace  # type: ignore[import]


class ServiceContextClass:
    def __init__(self) -> None:
        self.request_id_var: ContextVar[int | None] = ContextVar(
            "request_id_var", default=None
        )
        self.component_name_var: ContextVar[str] = ContextVar(
            "component_name", default="cli"
        )

    @property
    def sampled(self) -> int | None:
        span = trace.get_current_span()
        if span is None:
            return None
        return 1 if span.get_span_context().trace_flags.sampled else 0

    @property
    def trace_id(self) -> int | None:
        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().trace_id

    @property
    def span_id(self) -> int | None:
        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().span_id

    @property
    def request_id(self) -> int | None:
        return self.request_id_var.get()

    @property
    def component_name(self) -> str | None:
        return self.component_name_var.get()


ServiceContext = ServiceContextClass()
