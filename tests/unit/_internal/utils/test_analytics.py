from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from schema import Or
from schema import And
from schema import Schema
from _pytest.logging import LogCaptureFixture

import bentoml
from bentoml._internal.utils import analytics

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.monkeypatch import MonkeyPatch

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
    return analytics.schemas.ModelSaveEvent(
        module="test",
        model_size_in_kb=123123123,
    )


def test_get_payload(event_properties: analytics.schemas.ModelSaveEvent):
    payload = analytics.usage_stats.get_payload(
        event_properties=event_properties, session_id="random_session_id"
    )
    assert SCHEMA.validate(payload)


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_send_usage(
    mock_do_not_track: MagicMock,
    mock_post: MagicMock,
    event_properties: analytics.schemas.ModelSaveEvent,
):
    mock_do_not_track.return_value = False
    analytics.track(event_properties)
    assert mock_do_not_track.called
    assert mock_post.called


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@pytest.mark.usefixtures("event_properties")
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
@pytest.mark.usefixtures("event_properties")
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
    mock_logger.debug.assert_called_with("Tracking Error: something went wrong")


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@pytest.mark.parametrize("production", [False, True])
@pytest.mark.usefixtures("noop_service")
def test_track_serve_init(
    mock_post: MagicMock,
    mock_do_not_track: MagicMock,
    noop_service: bentoml.Service,
    production: bool,
):

    mock_do_not_track.return_value = False

    mock_response = Mock()
    mock_post.return_value = mock_response
    mock_response.text = "sent"

    analytics.usage_stats.track_serve_init(
        noop_service,
        production=production,
        serve_info=analytics.usage_stats.get_serve_info(),
    )

    assert not mock_do_not_track.called
    assert mock_post.called


@patch("bentoml._internal.server.metrics.prometheus.PrometheusClient")
@pytest.mark.parametrize(
    "mock_output",
    [
        b"",
        b"""# HELP BENTOML_noop_request_total Multiprocess metric""",
    ],
)
def test_get_metrics_report_filtered(
    mock_prometheus_client: MagicMock, mock_output: bytes
):
    mock_prometheus_client.multiproc.return_value = False
    mock_prometheus_client.generate_latest.return_value = mock_output
    assert analytics.usage_stats.get_metrics_report(mock_prometheus_client) == []


@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@pytest.mark.usefixtures("noop_service")
def test_track_serve_do_not_track(
    mock_do_not_track: MagicMock, noop_service: bentoml.Service
):
    mock_do_not_track.return_value = True
    with analytics.track_serve(
        noop_service,
        production=False,
        serve_info=analytics.usage_stats.get_serve_info(),
    ) as output:
        pass

    assert not output
    assert mock_do_not_track.called


@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.track_serve_init")
@patch("bentoml._internal.utils.analytics.usage_stats.usage_event_debugging")
@patch("bentoml._internal.server.metrics.prometheus.PrometheusClient")
@pytest.mark.usefixtures("propagate_logs")
@pytest.mark.usefixtures("noop_service")
def test_track_serve(
    mock_prometheus_client: MagicMock,
    mock_usage_event_debugging: MagicMock,
    mock_track_serve_init: MagicMock,
    mock_post: MagicMock,
    mock_do_not_track: MagicMock,
    noop_service: bentoml.Service,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
):
    mock_prometheus_client.multiproc.return_value = False
    mock_prometheus_client.generate_latest.return_value = b"""\
# HELP BENTOML_noop_request_total Multiprocess metric
# TYPE BENTOML_noop_request_total counter
BENTOML_noop_request_total{endpoint="/docs.json",http_response_code="200",service_version=""} 2.0
BENTOML_noop_request_total{endpoint="/classify",http_response_code="200",service_version=""} 9.0
BENTOML_noop_request_total{endpoint="/",http_response_code="200",service_version=""} 1.0
# HELP BENTOML_noop_request_in_progress Multiprocess metric
# TYPE BENTOML_noop_request_in_progress gauge
BENTOML_noop_request_in_progress{endpoint="/",service_version=""} 0.0
BENTOML_noop_request_in_progress{endpoint="/docs.json",service_version=""} 0.0
BENTOML_noop_request_in_progress{endpoint="/classify",service_version=""} 0.0
# HELP BENTOML_noop_request_duration_seconds Multiprocess metric
# TYPE BENTOML_noop_request_duration_seconds histogram
            """
    mock_do_not_track.return_value = False
    mock_usage_event_debugging.return_value = True

    monkeypatch.setenv("__BENTOML_DEBUG_USAGE", "True")
    analytics.usage_stats.SERVE_USAGE_TRACKING_INTERVAL_SECONDS = 1

    with caplog.at_level(logging.INFO):
        with analytics.track_serve(
            noop_service,
            production=False,
            metrics_client=mock_prometheus_client,
            serve_info=analytics.usage_stats.get_serve_info(),
        ):
            import time

            time.sleep(2)
            pass

    assert not mock_post.called
    assert mock_do_not_track.called
    assert mock_track_serve_init.called

    assert "Tracking Payload" in caplog.text
    assert not any(x in caplog.text for x in analytics.usage_stats.EXCLUDE_PATHS)
    assert "static_content" not in caplog.text
