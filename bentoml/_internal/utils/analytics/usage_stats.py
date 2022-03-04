import os
import typing as t
import logging
import secrets
import platform
import threading
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from functools import wraps
from functools import lru_cache

import yaml
import psutil
import requests
from simple_di import inject
from simple_di import Provide

from bentoml import __version__ as BENTOML_VERSION

from ...configuration.containers import CLIENT_ID_PATH
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

logger = logging.getLogger(__name__)

BENTOML_DO_NOT_TRACK = "BENTOML_DO_NOT_TRACK"
BENTOML_TRACKING_URL = "https://t.bentoml.com"
BENTOML_USAGE_REPORT_INTERVAL_SECONDS = "__BENTOML_USAGE_REPORT_INTERVAL_SECONDS"
BENTO_SERVE_SCHEDULED_TRACK_EVENT_TYPE = "bentoml_bento_serve_scheduled"


def slient(func: "t.Callable[P, T]") -> "t.Callable[P, T]":  # pragma: no cover
    # Slient errors when tracking
    @wraps(func)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            logger.debug(f"{err}")

    return wrapper


@lru_cache(maxsize=1)
def do_not_track() -> bool:
    # Returns True if and only if the environment variable is defined and has value True.
    # The function is cached for better performance.
    return os.environ.get(BENTOML_DO_NOT_TRACK, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def get_usage_stats_interval_seconds() -> int:
    # The interval for getting usage stats for a serve is 12 hours.
    # cached for better performance.
    return int(os.environ.get(BENTOML_USAGE_REPORT_INTERVAL_SECONDS, 12 * 60 * 60))


def get_serve_info() -> t.Dict[str, str]:  # pragma: no cover
    # Returns a safe token for serve as well as timestamp of creating this token
    return {
        "serve_id": secrets.token_urlsafe(32),
        "serve_creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@lru_cache(maxsize=1)
def get_platform() -> str:
    return platform.platform(aliased=True)


@lru_cache(maxsize=1)
def get_python_version() -> str:
    return platform.python_version()


@lru_cache(maxsize=1)
def get_client_id() -> t.Dict[str, str]:  # pragma: no cover
    if os.path.exists(CLIENT_ID_PATH):
        with open(CLIENT_ID_PATH, "r", encoding="utf-8") as f:
            client_id = yaml.safe_load(f)
        return client_id
    return {}


def get_hardware_usage(pid: int) -> t.Dict[str, t.Any]:  # pragma: no cover
    proc = psutil.Process(pid)
    with proc.oneshot():
        return {
            "memory_usage_percent": proc.memory_percent(),
            "total_memory(MB)": psutil.virtual_memory().total / 1024**2,
            "num_threads": proc.num_threads(),
        }


@inject
def get_payload(
    event_type: str,
    event_pid: int,
    event_properties: t.Optional[t.Dict[str, t.Any]] = None,
    session_id: str = Provide[BentoMLContainer.session_id],
) -> t.Dict[str, t.Any]:
    """
    event_type(str): type of event (model_save, bento_build, etc.)
    event_pid(int): retrieved with os.getpid() for the given event
    event_properties(dict(str, Any)): custom properties of a given event
    """
    if event_properties is None:
        event_properties = {}

    # TODO: add yatai_user email and yatai_version when available
    common_stats = {
        "bentoml_version": BENTOML_VERSION,
        "python_version": get_python_version(),
        "platform": get_platform(),
        "session_id": session_id,
        **get_client_id(),
    }

    return {
        "event_type": event_type,
        **event_properties,
        **common_stats,
        **get_hardware_usage(event_pid),
    }


@slient
def send_usage_event(payload: t.Dict[str, t.Any], timeout: int, uri: str) -> None:
    requests.post(uri, json=payload, timeout=timeout)


@slient
def track(
    event_type: str,
    event_pid: int,
    event_properties: t.Optional[t.Dict[str, t.Any]] = None,
    timeout: int = 2,
    uri: str = BENTOML_TRACKING_URL,
) -> None:
    if do_not_track():
        return
    payload = get_payload(
        event_type=event_type, event_pid=event_pid, event_properties=event_properties
    )
    send_usage_event(payload, timeout=timeout, uri=uri)


@slient
def scheduled_track(
    event_pid: int,
    event_properties: t.Optional[t.Dict[str, t.Any]] = None,
    interval: int = get_usage_stats_interval_seconds(),
) -> t.Tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()

    def loop() -> t.NoReturn:  # type: ignore
        while not stop_event.wait(interval):
            track(
                BENTO_SERVE_SCHEDULED_TRACK_EVENT_TYPE,
                event_pid,
                event_properties=event_properties,
            )

    thread = threading.Thread(target=loop, daemon=True)
    return thread, stop_event
