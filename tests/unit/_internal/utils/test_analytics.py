# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from schema import Or
from schema import And
from schema import Schema
from prometheus_client.parser import (
    text_string_to_metric_families,  # type: ignore (no prometheus types)
)

import bentoml
from bentoml._internal.utils import analytics

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from prometheus_client.metrics_core import Metric

    from bentoml import Service

SCHEMA = Schema(
    {
        "common_properties": {
            "timestamp": str,
            "bentoml_version": str,
            "client": {"creation_timestamp": str, "id": str},
            "memory_usage_percent": Or(int, float),
            "platform": str,
            "python_version": str,
            "total_memory_in_mb": int,
            "yatai_user_email": Or(str, None),
            "yatai_version": Or(str, None),
            "yatai_org_uid": Or(str, None),
            "yatai_cluster_uid": Or(str, None),
            "yatai_deployment_uid": Or(str, None),
            "is_interactive": bool,
            "in_notebook": bool,
        },
        "event_properties": {
            "module": str,
            "model_size_in_kb": Or(float, int),
        },
        "session_id": str,
        "event_type": And(str, str.islower),
    }
)


@pytest.fixture(scope="function", name="event_properties")
def fixture_event_properties() -> analytics.schemas.ModelSaveEvent:
    return analytics.schemas.ModelSaveEvent(module="test", model_size_in_kb=123123123)


def test_get_payload(event_properties: analytics.schemas.ModelSaveEvent):
    payload = analytics.usage_stats.get_payload(
        event_properties=event_properties, session_id="random_session_id"
    )
    assert SCHEMA.validate(payload)


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@patch("bentoml._internal.utils.analytics.usage_stats._usage_event_debugging")
def test_send_usage(
    mock_usage_event_debugging: MagicMock,
    mock_do_not_track: MagicMock,
    mock_post: MagicMock,
    event_properties: analytics.schemas.ModelSaveEvent,
    caplog: LogCaptureFixture,
):
    mock_usage_event_debugging.return_value = False
    mock_do_not_track.return_value = False
    analytics.track(event_properties)

    assert mock_do_not_track.called
    assert mock_post.called

    mock_usage_event_debugging.return_value = True
    with caplog.at_level(logging.INFO):
        analytics.track(event_properties)

    assert "Tracking Payload" in caplog.text


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_do_not_track(
    mock_do_not_track: MagicMock,
    mock_post: MagicMock,
    event_properties: analytics.schemas.ModelSaveEvent,
):
    mock_do_not_track.return_value = True
    analytics.track(event_properties)

    assert mock_do_not_track.called
    assert not mock_post.called


@patch("bentoml._internal.utils.analytics.usage_stats.logger")
@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_send_usage_failure(
    mock_do_not_track: MagicMock,
    mock_post: MagicMock,
    mock_logger: MagicMock,
    event_properties: analytics.schemas.ModelSaveEvent,
):
    mock_do_not_track.return_value = False
    mock_post.side_effect = AssertionError("something went wrong")
    # nothing should happen
    analytics.track(event_properties)
    assert mock_do_not_track.called
    assert mock_post.called
    mock_logger.debug.assert_called_with("Tracking Error: %s", mock_post.side_effect)


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@patch("bentoml._internal.utils.analytics.usage_stats._usage_event_debugging")
@pytest.mark.parametrize("production", [False, True])
@pytest.mark.usefixtures("propagate_logs")
def test_track_serve_init(
    mock_usage_event_debugging: MagicMock,
    mock_do_not_track: MagicMock,
    mock_post: MagicMock,
    simple_service: Service,
    production: bool,
    caplog: LogCaptureFixture,
):

    mock_do_not_track.return_value = False
    mock_usage_event_debugging.return_value = False

    mock_response = Mock()
    mock_post.return_value = mock_response
    mock_response.text = "sent"

    analytics.usage_stats._track_serve_init(  # type: ignore (private warning)
        simple_service,
        production=production,
        serve_info=analytics.usage_stats.get_serve_info(),
        serve_kind="http",
    )

    assert mock_do_not_track.called
    assert mock_post.called

    mock_usage_event_debugging.return_value = True
    with caplog.at_level(logging.INFO):
        analytics.usage_stats._track_serve_init(  # type: ignore (private warning)
            simple_service,
            production=production,
            serve_info=analytics.usage_stats.get_serve_info(),
            serve_kind="http",
        )
    assert "model_types" in caplog.text


@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@patch("bentoml._internal.utils.analytics.usage_stats._usage_event_debugging")
def test_track_serve_init_no_bento(
    mock_usage_event_debugging: MagicMock,
    mock_do_not_track: MagicMock,
    caplog: LogCaptureFixture,
):
    logger = logging.getLogger("bentoml")
    logger.propagate = False

    mock_do_not_track.return_value = False
    mock_usage_event_debugging.return_value = True
    caplog.clear()

    with caplog.at_level(logging.INFO):
        analytics.usage_stats._track_serve_init(  # type: ignore (private warning)
            bentoml.Service("test"),
            production=False,
            serve_info=analytics.usage_stats.get_serve_info(),
            serve_kind="http",
        )
    assert "model_types" not in caplog.text


@patch("bentoml._internal.server.metrics.prometheus.PrometheusClient")
@pytest.mark.parametrize(
    "mock_output,expected",
    [
        (b"", []),
        (
            b"""# HELP BENTOML_noop_request_total Multiprocess metric""",
            [],
        ),
    ],
)
@pytest.mark.parametrize("serve_kind", ["grpc", "http"])
def test_filter_metrics_report(
    mock_prometheus_client: MagicMock,
    mock_output: bytes,
    expected: tuple[list[t.Any], bool | None],
    serve_kind: str,
):
    mock_prometheus_client.multiproc.return_value = False
    mock_prometheus_client.generate_latest.return_value = mock_output
    assert (
        analytics.usage_stats.get_metrics_report(
            mock_prometheus_client, serve_kind=serve_kind
        )
        == expected
    )


@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_track_serve_do_not_track(
    mock_do_not_track: MagicMock, simple_service: Service
):
    mock_do_not_track.return_value = True
    with analytics.track_serve(
        simple_service,
        production=False,
        serve_info=analytics.usage_stats.get_serve_info(),
    ) as output:
        pass

    assert not output
    assert mock_do_not_track.called


@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@patch("bentoml._internal.server.metrics.prometheus.PrometheusClient")
def test_legacy_get_metrics_report(
    mock_prometheus_client: MagicMock,
    mock_do_not_track: MagicMock,
    simple_service: Service,
):
    mock_do_not_track.return_value = True
    mock_prometheus_client.multiproc.return_value = False
    mock_prometheus_client.text_string_to_metric_families.return_value = text_string_to_metric_families(
        b"""\
# HELP BENTOML_simple_service_request_in_progress Multiprocess metric
# TYPE BENTOML_simple_service_request_in_progress gauge
BENTOML_simple_service_request_in_progress{endpoint="/predict",service_version="not available"} 0.0
# HELP BENTOML_simple_service_request_total Multiprocess metric
# TYPE BENTOML_simple_service_request_total counter
BENTOML_simple_service_request_total{endpoint="/predict",http_response_code="200",service_version="not available"} 8.0
""".decode(
            "utf-8"
        )
    )
    output = analytics.usage_stats.get_metrics_report(
        mock_prometheus_client, serve_kind="http"
    )
    assert {
        "endpoint": "/predict",
        "http_response_code": "200",
        "service_version": "not available",
        "value": 8.0,
    } in output

    endpoints = [filtered["endpoint"] for filtered in output]

    assert not any(x in endpoints for x in analytics.usage_stats.EXCLUDE_PATHS)


@patch("bentoml._internal.server.metrics.prometheus.PrometheusClient")
@pytest.mark.parametrize(
    "serve_kind,expected",
    [
        (
            "grpc",
            {
                "api_name": "pred_json",
                "http_response_code": "200",
                "service_name": "simple_service",
                "service_version": "not available",
                "value": 15.0,
            },
        ),
        ("http", None),
    ],
)
@pytest.mark.parametrize(
    "generated_metrics",
    [
        text_string_to_metric_families(
            b"""\
                # HELP bentoml_api_server_request_total Multiprocess metric
                # TYPE bentoml_api_server_request_total counter
                bentoml_api_server_request_total{api_name="pred_json",http_response_code="200",service_name="simple_service",service_version="not available"} 15.0
                # HELP bentoml_api_server_request_in_progress Multiprocess metric
                # TYPE bentoml_api_server_request_in_progress gauge
                bentoml_api_server_request_in_progress{api_name="pred_json",service_name="simple_service",service_version="not available"} 0.0
                """.decode(
                "utf-8"
            )
        )
    ],
)
def test_get_metrics_report(
    mock_prometheus_client: MagicMock,
    simple_service: Service,
    serve_kind: str,
    expected: dict[str, str | float] | None,
    generated_metrics: t.Generator[Metric, None, None],
):
    mock_prometheus_client.multiproc.return_value = False
    mock_prometheus_client.text_string_to_metric_families.return_value = (
        generated_metrics
    )
    output = analytics.usage_stats.get_metrics_report(
        mock_prometheus_client, serve_kind=serve_kind
    )
    if expected:
        assert expected in output


@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats._track_serve_init")
@patch("bentoml._internal.utils.analytics.usage_stats._usage_event_debugging")
@patch("bentoml._internal.server.metrics.prometheus.PrometheusClient")
@pytest.mark.usefixtures("propagate_logs")
def test_track_serve(
    mock_prometheus_client: MagicMock,
    mock_usage_event_debugging: MagicMock,
    mock_track_serve_init: MagicMock,
    mock_post: MagicMock,
    mock_do_not_track: MagicMock,
    simple_service: Service,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
):
    mock_prometheus_client.multiproc.return_value = False
    mock_do_not_track.return_value = False
    mock_usage_event_debugging.return_value = True

    monkeypatch.setenv("__BENTOML_DEBUG_USAGE", "True")
    analytics.usage_stats.SERVE_USAGE_TRACKING_INTERVAL_SECONDS = 1

    with caplog.at_level(logging.INFO):
        with analytics.track_serve(
            simple_service,
            production=False,
            metrics_client=mock_prometheus_client,
            serve_info=analytics.usage_stats.get_serve_info(),
        ):
            import time

            time.sleep(2)

    assert not mock_post.called
    assert mock_do_not_track.called
    assert mock_track_serve_init.called
