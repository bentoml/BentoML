from __future__ import annotations

from uuid import uuid4
from typing import TYPE_CHECKING
from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from schema import Or
from schema import And
from schema import Schema

import bentoml
from bentoml._internal.utils import analytics

if TYPE_CHECKING:
    from unittest.mock import MagicMock

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
@pytest.mark.parametrize(
    "production, serve_info", [(False, uuid4().hex), (True, uuid4().hex)]
)
@pytest.mark.usefixtures("noop_service")
def test_track_serve_init(
    mock_post: MagicMock,
    mock_do_not_track: MagicMock,
    noop_service: bentoml.Service,
    production: bool,
    serve_info: str,
):
    from bentoml._internal.utils.analytics.usage_stats import ServeInfo

    mock_do_not_track.return_value = False

    mock_response = Mock()
    mock_post.return_value = mock_response
    mock_response.text = "sent"

    analytics.usage_stats.track_serve_init(
        noop_service,
        production=production,
        serve_info=ServeInfo(
            serve_id=serve_info, serve_started_timestamp=datetime.now()
        ),
    )

    assert not mock_do_not_track.called
    assert mock_post.called
