from datetime import datetime
import typing as t
from typing import TYPE_CHECKING

import requests
from schema import Or
from schema import And
from schema import Schema

import bentoml._internal.utils.analytics as analytics_lib
from bentoml._internal.utils.analytics.usage_stats import ServeInfo

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

_is_lower: t.Callable[[str], bool] = lambda s: s.islower()

EVENT_TYPE = "bentoml_test_event"
EVENT_PID = 123
SESSION_ID = "asdfasdf"

SCHEMA = Schema(
    {
        "common_properties": {
            "bentoml_version": str,
            "client_id": {
            "client_creation_timestamp": str,
            "client_id": str 
            },
            "memory_usage_percent": Or(int, float),
            "num_threads": int,
            "platform": str,
            "python_version": str,
            "total_memory_in_kb": Or(int, float),
            "yatai_user_email": str
        },
        "event_properties": {
            "bento_identifier": str,
            "serve_creation_timestamp": str,
            "serve_id": str,
            "serve_location": str
        },
        "session_id": str,
        "event_type": And(str, _is_lower),
    }
)


def patch_get_client_id():
    return {
        "client_id": "481371785bd475833ead69073bf081b860f20b2a0ed423a7d0eaed734e5a0d8a",
        "client_creation_timestamp": "2222-02-28T06:06:23.798993+00:00",
    }

def patch_get_serve_info():
    return ServeInfo(serve_id="1324",serve_creation_timestamp=datetime.fromisoformat("2222-02-28T06:06:23.798993+00:00"))


def patch_get_hardware_usage(pid: int):
    return {
        "memory_usage_percent": 0.22068023681640625,
        "total_memory(MB)": 32768,
        "num_threads": 1,
    }


def test_get_payload(monkeypatch: "MonkeyPatch"):
    with monkeypatch.context() as m:
        m.setattr(analytics_lib, "get_client_id", patch_get_client_id)
        m.setattr(analytics_lib, "get_hardware_usage",
                  patch_get_hardware_usage)
        payload = analytics_lib.get_payload(
            EVENT_TYPE, event_pid=EVENT_PID, session_id=SESSION_ID
        )
        assert SCHEMA.validate(payload)


def test_do_not_track(monkeypatch: "MonkeyPatch"):
    with monkeypatch.context() as m:
        m.setenv("BENTOML_DO_NOT_TRACK", "True")
        assert analytics_lib.do_not_track() is True


def test_send_usage_event(monkeypatch: "MonkeyPatch"):
    def patch_post(*args: t.Any, **kwargs: t.Any) -> t.Dict[str, str]:
        return {"Hello": "World"}

    with monkeypatch.context() as m:
        m.setattr(
            analytics_lib, "BENTOML_TRACKING_URL", "http://127.0.0.1:8000/tracking"
        )
        m.setattr(analytics_lib, "BENTOML_USAGE_REPORT_INTERVAL_SECONDS", "1")
        m.setattr(analytics_lib, "get_client_id", patch_get_client_id)
        m.setattr(analytics_lib, "get_hardware_usage",
                  patch_get_hardware_usage)
        m.setattr(requests, "post", patch_post)
        analytics_lib.send_usage_event(
            analytics_lib.get_payload(EVENT_TYPE, EVENT_PID),
            2,
            analytics_lib.BENTOML_TRACKING_URL,
        )
        analytics_lib.track(EVENT_TYPE, EVENT_PID)
