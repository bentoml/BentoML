from typing import TYPE_CHECKING
from unittest.mock import patch
import requests

from schema import Or
from schema import And
from schema import Schema

import bentoml._internal.utils.analytics as analytics_lib

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

SCHEMA = Schema(
    {
        "common_properties": {
            "timestamp": str,
            "bentoml_version": str,
            "client": {"creation_timestamp": str, "id": str},
            "memory_usage_percent": Or(int, float),
            "num_threads": int,
            "platform": str,
            "python_version": str,
            "total_memory_in_kb": Or(int, float),
            "yatai_user_email": Or(str, None),
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


def test_get_payload():
    event_properties = analytics_lib.schemas.ModelSaveEvent(
        module="test",
        model_size_in_kb=123123123,
    )
    payload = analytics_lib.usage_stats.get_payload(
        event_properties=event_properties, session_id="random_session_id"
    )
    assert SCHEMA.validate(payload)


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_send_usage(mock_do_not_track, mock_post):
    event_properties = analytics_lib.schemas.ModelSaveEvent(
        module="test",
        model_size_in_kb=123123123,
    )

    mock_do_not_track.return_value = False
    analytics_lib.track(
        event_properties,
    )
    assert mock_do_not_track.called
    assert mock_post.called


@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_do_not_track(mock_do_not_track, mock_post):
    event_properties = analytics_lib.schemas.ModelSaveEvent(
        module="test",
        model_size_in_kb=123123123,
    )

    mock_do_not_track.return_value = True
    analytics_lib.track(
        event_properties,
    )
    assert mock_do_not_track.called
    assert not mock_post.called


@patch("bentoml._internal.utils.analytics.usage_stats.logger")
@patch("bentoml._internal.utils.analytics.usage_stats.requests.post")
@patch("bentoml._internal.utils.analytics.usage_stats.do_not_track")
def test_send_usage_failure(mock_do_not_track, mock_post, mock_logger):
    event_properties = analytics_lib.schemas.ModelSaveEvent(
        module="test",
        model_size_in_kb=123123123,
    )

    mock_do_not_track.return_value = False
    mock_post.side_effect = AssertionError('something went wrong')
    # nothing should happen
    analytics_lib.track(
        event_properties,
    )
    assert mock_do_not_track.called
    assert mock_post.called
    mock_logger.debug.assert_called_with("something went wrong")