import os
import typing as t
import logging
import secrets
import threading
import contextlib
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
    serve_started_timestamp: datetime


def get_serve_info() -> ServeInfo:  # pragma: no cover
    # Returns a safe token for serve as well as timestamp of creating this token
    return ServeInfo(
        serve_id=secrets.token_urlsafe(32),
        serve_started_timestamp=datetime.now(timezone.utc),
    )


@inject
def get_payload(
    event_properties: EventMeta,
    session_id: str = Provide[BentoMLContainer.session_id],
    injected_payload: t.Optional[t.Dict[str, t.Dict[str, t.Any]]] = None,
) -> t.Dict[str, t.Any]:
    payload = TrackingPayload(
        session_id=session_id,
        common_properties=CommonProperties(),
        event_properties=event_properties,
    )
    res = asdict(payload, value_serializer=serializer)
    if injected_payload is not None:
        for k, v in injected_payload.items():
            res[k].update(v)
    return res


@slient
def track(
    event_properties: EventMeta,
    *,
    injected_payload: t.Optional[t.Dict[str, t.Dict[str, t.Any]]] = None,
    uri: str = BENTOML_TRACKING_URL,
    timeout: int = 2,
) -> t.Optional[requests.Response]:
    if do_not_track():
        return
    payload = get_payload(
        event_properties=event_properties, injected_payload=injected_payload
    )
    return requests.post(uri, json=payload, timeout=timeout)


@contextlib.contextmanager
def scheduled_track(
    event_properties: EventMeta,
    interval: int = BENTOML_USAGE_REPORT_INTERVAL_SECONDS,
):  # pragma: no cover
    if do_not_track():
        yield
        return

    stop_event = threading.Event()

    def loop() -> t.NoReturn:  # type: ignore
        while not stop_event.wait(interval):
            track(event_properties=event_properties)

    tracking_thread = threading.Thread(target=loop, daemon=True)
    try:
        tracking_thread.start()
        yield
    finally:
        tracking_thread.join()
