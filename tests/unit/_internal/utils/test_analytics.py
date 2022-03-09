import typing as t
from typing import TYPE_CHECKING
from datetime import datetime

import requests
from schema import Or
from schema import And
from schema import Schema

import bentoml._internal.utils.analytics as analytics_lib
from bentoml._internal.types import Tag

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

_is_lower: t.Callable[[str], bool] = lambda s: s.islower()

SESSION_ID = "asdfasdf"

SCHEMA = Schema(
    {
        "common_properties": {
            "bentoml_version": str,
            "client_id": {"client_creation_timestamp": str, "client_id": str},
            "memory_usage_percent": Or(int, float),
            "num_threads": int,
            "platform": str,
            "python_version": str,
            "total_memory_in_kb": Or(int, float),
            "yatai_user_email": str,
        },
        "event_properties": {
            "module": str,
            "model_tag": str,
            "model_creation_timestamp": str,
            "model_size_in_kb": Or(float, int),
        },
        "session_id": str,
        "event_type": And(str, _is_lower),
    }
)


def test_get_payload():
    event_properties = analytics_lib.schemas.ModelSaveEvent(
        module="test",
        model_tag=Tag("test"),
        model_creation_timestamp=datetime.fromisoformat(
            "2222-02-28T06:06:23.798993+00:00"
        ),
        model_size_in_kb=123123123,
    )
    payload = analytics_lib.usage_stats.get_payload(
        event_properties=event_properties, session_id=SESSION_ID
    )
    assert SCHEMA.validate(payload)


def test_do_not_track(monkeypatch: "MonkeyPatch"):
    with monkeypatch.context() as m:
        m.setenv("BENTOML_DO_NOT_TRACK", "True")
        assert analytics_lib.usage_stats.do_not_track() is True


def test_send_usage_event(monkeypatch: "MonkeyPatch"):
    def patch_post(*args: t.Any, **kwargs: t.Any) -> t.Dict[str, str]:
        return {"Hello": "World"}

    with monkeypatch.context() as m:
        m.setattr(
            analytics_lib.usage_stats,
            "BENTOML_TRACKING_URL",
            "http://127.0.0.1:8000/tracking",
        )
        m.setattr(
            analytics_lib.usage_stats, "BENTOML_USAGE_REPORT_INTERVAL_SECONDS", "1"
        )
        m.setattr(requests, "post", patch_post)
        event_properties = analytics_lib.schemas.ModelSaveEvent(
            module="test",
            model_tag=Tag("test"),
            model_creation_timestamp=datetime.fromisoformat(
                "2222-02-28T06:06:23.798993+00:00"
            ),
            model_size_in_kb=123123123,
        )
        payload = analytics_lib.usage_stats.get_payload(
            event_properties=event_properties, session_id=SESSION_ID
        )
        r = analytics_lib.usage_stats.send_usage_event(
            payload,
            uri=analytics_lib.usage_stats.BENTOML_TRACKING_URL,
            timeout=2,
        )
        assert r == {"Hello": "World"}
        analytics_lib.track(
            event_properties=event_properties,
            uri=analytics_lib.usage_stats.BENTOML_TRACKING_URL,
        )
