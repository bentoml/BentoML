from __future__ import annotations

import os
from unittest import mock

import pytest

from bentoml._internal.monitoring.otlp import OTLPMonitor

OTEL_EXPORTER_OTLP_HEADERS = "OTEL_EXPORTER_OTLP_HEADERS"


def test_otlp_monitor_init_logger_with_dict_headers_http(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(OTEL_EXPORTER_OTLP_HEADERS, raising=False)

    headers = {"Authorization": "Bearer test_token", "X-Custom-Header": "custom_value"}
    monitor = OTLPMonitor(
        name="test_otlp_monitor_http_dict",
        endpoint="http://localhost:4318",
        headers=headers,
        protocol="http",
    )

    with mock.patch(
        "bentoml._internal.monitoring.otlp.OTLPHttpLogExporter"
    ) as mock_exporter:
        monitor._init_logger()
        mock_exporter.assert_called_once_with(
            endpoint="http://localhost:4318",
            certificate_file=None,
            headers=headers,
            timeout=None,
            compression=None,
        )

    assert os.environ.get(OTEL_EXPORTER_OTLP_HEADERS) is None


def test_otlp_monitor_init_logger_with_dict_headers_grpc(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(OTEL_EXPORTER_OTLP_HEADERS, raising=False)

    headers = {"Authorization": "Bearer grpc_token"}
    monitor = OTLPMonitor(
        name="test_otlp_monitor_grpc_dict",
        endpoint="http://localhost:4317",
        headers=headers,
        protocol="grpc",
    )

    with mock.patch(
        "bentoml._internal.monitoring.otlp.OTLPGrpcLogExporter"
    ) as mock_exporter:
        monitor._init_logger()
        mock_exporter.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=None,
            credentials=None,
            headers=headers,
            timeout=None,
            compression=None,
        )

    assert os.environ.get(OTEL_EXPORTER_OTLP_HEADERS) is None


def test_otlp_monitor_init_logger_with_string_headers_http(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(OTEL_EXPORTER_OTLP_HEADERS, raising=False)

    monitor = OTLPMonitor(
        name="test_otlp_monitor_http_str",
        endpoint="http://localhost:4318",
        headers=" Authorization = Bearer test_token , X-Custom = val=extra ",
        protocol="http",
    )

    with mock.patch(
        "bentoml._internal.monitoring.otlp.OTLPHttpLogExporter"
    ) as mock_exporter:
        monitor._init_logger()
        mock_exporter.assert_called_once_with(
            endpoint="http://localhost:4318",
            certificate_file=None,
            headers={"Authorization": "Bearer test_token", "X-Custom": "val=extra"},
            timeout=None,
            compression=None,
        )

    assert os.environ.get(OTEL_EXPORTER_OTLP_HEADERS) is None


def test_otlp_monitor_init_logger_with_string_headers_grpc(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(OTEL_EXPORTER_OTLP_HEADERS, raising=False)

    monitor = OTLPMonitor(
        name="test_otlp_monitor_grpc_str",
        endpoint="http://localhost:4317",
        headers="Authorization=Bearer grpc_token,X-Custom=val",
        protocol="grpc",
    )

    with mock.patch(
        "bentoml._internal.monitoring.otlp.OTLPGrpcLogExporter"
    ) as mock_exporter:
        monitor._init_logger()
        mock_exporter.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=None,
            credentials=None,
            headers={"Authorization": "Bearer grpc_token", "X-Custom": "val"},
            timeout=None,
            compression=None,
        )

    assert os.environ.get(OTEL_EXPORTER_OTLP_HEADERS) is None
