from __future__ import annotations

import collections
from unittest import mock

import pytest

from bentoml._internal.monitoring.default import DefaultMonitor


def _assert_trace_context_columns(monitor, context_path: str) -> None:
    monitor.data_logger = mock.Mock()
    with mock.patch(context_path) as context:
        context.request_id = 101
        context.trace_id = 202
        context.span_id = 303
        context.service_name = "fraud-service"
        monitor.export_data({"prediction": collections.deque(["fraud"])})

    record = monitor.data_logger.info.call_args.args[0]
    assert record["request_id"] == "101"
    assert record["trace_id"] == "202"
    assert record["span_id"] == "303"
    assert record["service_name"] == "fraud-service"


def test_default_monitor_includes_span_and_service_context(tmp_path) -> None:
    monitor = DefaultMonitor("predictions", str(tmp_path))
    _assert_trace_context_columns(
        monitor, "bentoml._internal.monitoring.default.trace_context"
    )


def test_otlp_monitor_includes_span_and_service_context() -> None:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http._log_exporter")
    from bentoml._internal.monitoring.otlp import OTLPMonitor

    monitor = OTLPMonitor("predictions", meta_sample_rate=0)
    _assert_trace_context_columns(
        monitor, "bentoml._internal.monitoring.otlp.trace_context"
    )
