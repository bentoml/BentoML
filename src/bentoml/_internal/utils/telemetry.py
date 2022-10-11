# type: ignore[reportMissingTypeStubs]
import typing as t
from typing import TYPE_CHECKING

from opentelemetry.trace import get_current_span
from opentelemetry.sdk.trace.sampling import Decision
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.sdk.trace.sampling import ParentBased
from opentelemetry.sdk.trace.sampling import StaticSampler
from opentelemetry.sdk.trace.sampling import SamplingResult
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

if TYPE_CHECKING:
    from opentelemetry.trace import Link
    from opentelemetry.trace import Context
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace import TraceState
    from opentelemetry.util.types import Attributes


def _get_parent_trace_state(parent_context: "Context") -> t.Optional["TraceState"]:
    parent_span_context = get_current_span(parent_context).get_span_context()
    if parent_span_context is None or not parent_span_context.is_valid:
        return None
    return parent_span_context.trace_state


class TraceIdRatioBasedAlwaysRecording(TraceIdRatioBased):
    """
    A trace Sampler that:
    * always recording (so that we can get the trace_id)
    * respect the parent's trace_state
    * ratio sampling
    """

    def should_sample(
        self,
        parent_context: t.Optional["Context"],
        trace_id: int,
        name: str,
        kind: t.Optional["SpanKind"] = None,
        attributes: t.Optional["Attributes"] = None,
        links: t.Optional[t.Sequence["Link"]] = None,
        trace_state: t.Optional["TraceState"] = None,
    ) -> "SamplingResult":
        decision = Decision.RECORD_ONLY
        if trace_id & self.TRACE_ID_LIMIT < self.bound:
            decision = Decision.RECORD_AND_SAMPLE
        if decision is Decision.RECORD_ONLY:
            pass
            # attributes = None
        return SamplingResult(
            decision,
            attributes,
            _get_parent_trace_state(parent_context),  # type: ignore[reportGeneralTypeIssues]
        )


class ParentBasedTraceIdRatio(ParentBased):
    """
    Sampler that respects its parent span's sampling decision, but otherwise
    samples probabalistically based on `rate`.
    """

    def __init__(self, rate: float):
        root = TraceIdRatioBasedAlwaysRecording(rate=rate)
        super().__init__(
            root=root,
            remote_parent_sampled=ALWAYS_ON,
            remote_parent_not_sampled=StaticSampler(Decision.RECORD_ONLY),
            local_parent_sampled=ALWAYS_ON,
            local_parent_not_sampled=StaticSampler(Decision.RECORD_ONLY),
        )
