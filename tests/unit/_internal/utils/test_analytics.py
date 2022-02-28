import os
import typing as t

import pytest
import requests
from schema import Or
from schema import And
from schema import Schema

import bentoml._internal.utils.analytics.usage_stats as analytics_lib

_is_lower: t.Callable[[str], bool] = lambda s: s.islower()
EVENT_TYPE = "bentoml_test_event"
EVENT_PID = 123
SESSION_ID = "asdfasdf"

SCHEMA = Schema(
    {
        "event_type": And(str, _is_lower),
        "bentoml_version": str,
        "python_version": str,
        "platform": str,
        "session_id": str,
        "client_creation_timestamp": str,
        "client_id": str,
        "memory_usage_percent": Or(int, float),
        "total_memory(MB)": Or(int, float),
        "num_threads": int,
    }
)


def patch_get_client_id():
    return {
        "client_id": "481371785bd475833ead69073bf081b860f20b2a0ed423a7d0eaed734e5a0d8a",
        "client_creation_timestamp": "2222-02-30T06:06:23.798993+00:00",
    }


def patch_get_hardware_usage(pid):
    return {
        "memory_usage_percent": 0.22068023681640625,
        "total_memory(MB)": 32768,
        "num_threads": 1,
    }


def test_get_payload(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(analytics_lib, "get_client_id", patch_get_client_id)
        m.setattr(analytics_lib, "get_hardware_usage", patch_get_hardware_usage)
        payload = analytics_lib.get_payload(
            EVENT_TYPE, event_pid=EVENT_PID, session_id=SESSION_ID
        )
        assert SCHEMA.validate(payload)


def test_do_not_track(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("BENTOML_DO_NOT_TRACK", "True")
        assert analytics_lib.do_not_track() is True


def test_send_usage_event(monkeypatch):
    def patch_post(*args, **kwargs):
        return {"Hello": "World"}

    with monkeypatch.context() as m:
        m.setattr(
            analytics_lib, "BENTOML_TRACKING_URL", "http://127.0.0.1:8000/tracking"
        )
        m.setenv("__BENTOML_USAGE_REPORT_INTERVAL_SECONDS", "1")
        m.setattr(analytics_lib, "get_client_id", patch_get_client_id)
        m.setattr(analytics_lib, "get_hardware_usage", patch_get_hardware_usage)
        m.setattr(requests, "post", patch_post)
        analytics_lib.send_usage_event(
            analytics_lib.get_payload(EVENT_TYPE, EVENT_PID),
            2,
            analytics_lib.BENTOML_TRACKING_URL,
        )
        analytics_lib.track(EVENT_TYPE, EVENT_PID)
