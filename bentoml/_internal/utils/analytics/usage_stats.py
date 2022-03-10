import os
import typing as t
import logging
import secrets
import threading
from typing import TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from datetime import timezone
from functools import wraps
from functools import lru_cache

import attrs
import requests
from attrs import asdict
from attrs import Attribute
from simple_di import inject
from simple_di import Provide

from ...types import Tag
from .schemas import EventMeta
from .schemas import TrackingPayload
from .schemas import CommonProperties
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

logger = logging.getLogger(__name__)

BENTOML_DO_NOT_TRACK = "BENTOML_DO_NOT_TRACK"
BENTOML_TRACKING_URL = "https://t.bentoml.com"
BENTOML_USAGE_REPORT_INTERVAL_SECONDS = int(12 * 60 * 60)


def slient(func: "t.Callable[P, T]") -> "t.Callable[P, T]":  # pragma: no cover
    # Slient errors when tracking
    @wraps(func)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            logger.debug(f"{err}")

    return wrapper


def serializer(
    inst: type, field: "Attribute[t.Any]", value: t.Any
) -> t.Any:  # pragma: no cover
    if isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, Tag):
        return str(value)
    elif isinstance(value, Path):
        return value.as_posix()
    else:
        return value


@lru_cache(maxsize=1)
def do_not_track() -> bool:
    # Returns True if and only if the environment variable is defined and has value True.
    # The function is cached for better performance.
    return os.environ.get(BENTOML_DO_NOT_TRACK, str(False)).lower() == "true"


@attrs.define
class ServeInfo:
    serve_id: str
    serve_creation_timestamp: datetime


def get_serve_info() -> ServeInfo:  # pragma: no cover
    # Returns a safe token for serve as well as timestamp of creating this token
    return ServeInfo(
        serve_id=secrets.token_urlsafe(32),
        serve_creation_timestamp=datetime.now(timezone.utc),
    )


@inject
def get_payload(
    event_properties: EventMeta,
    session_id: str = Provide[BentoMLContainer.session_id],
) -> t.Dict[str, t.Any]:
    payload = TrackingPayload(
        session_id=session_id,
        common_properties=CommonProperties(),
        event_properties=event_properties,
    )
    return asdict(payload, value_serializer=serializer)


@slient
def send_usage_event(
    payload: t.Dict[str, t.Any], uri: str, timeout: int
) -> requests.Response:
    return requests.post(uri, json=payload, timeout=timeout)


@slient
def track(
    event_properties: EventMeta,
    uri: str = BENTOML_TRACKING_URL,
    timeout: int = 2,
) -> None:
    if do_not_track():
        return
    send_usage_event(
        get_payload(event_properties=event_properties), timeout=timeout, uri=uri
    )


@slient
def scheduled_track(
    event_properties: EventMeta,
    interval: int = BENTOML_USAGE_REPORT_INTERVAL_SECONDS,
) -> t.Tuple[threading.Thread, threading.Event]:  # pragma: no cover
    stop_event = threading.Event()

    def loop() -> t.NoReturn:  # type: ignore
        while not stop_event.wait(interval):
            track(event_properties=event_properties)

    thread = threading.Thread(target=loop, daemon=True)
    return thread, stop_event
