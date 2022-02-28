import os
import typing as t
import asyncio
import logging
import secrets
import platform
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from functools import wraps
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import yaml
import psutil
import requests
from simple_di import inject
from simple_di import Provide

from bentoml import __version__ as BENTOML_VERSION
from bentoml.exceptions import BentoMLException

from ...configuration.containers import CLIENT_ID_PATH
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

logger = logging.getLogger(__name__)

TRACKING_URI = "https://t.bentoml.com"
BENTOML_DO_NOT_TRACK = "BENTOML_DO_NOT_TRACK"
BENTO_SERVE_SCHEDULED_TRACK_EVENT_TYPE = "bentoml_bento_serve_scheduled"


def experimental(func: "t.Callable[P, T]") -> "t.Callable[P, T]":
    experimental_doc = (
        "    .. note:: This function is experimental. "
        + "Use with care, as API might change in the future\n\n"
    )
    func.__doc__ = "" if func.__doc__ is None else func.__doc__
    func.__doc__ = experimental_doc + func.__doc__
    logger.debug(f"{func.__qualname__} is an experimental API. Use with care!")
    return func


def slient(func: "t.Callable[P, T]") -> "t.Callable[P, T]":
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
    # The interval for getting usage stats for a serve is 4 hours.
    # cached for better performance.
    return int(os.environ.get("__BENTOML_USAGE_REPORT_INTERVAL_SECONDS", 4 * 60 * 60))


def get_serve_info() -> t.Dict[str, str]:
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
def get_client_id() -> t.Dict[str, str]:
    if os.path.exists(CLIENT_ID_PATH):
        with open(CLIENT_ID_PATH, "r", encoding="utf-8") as f:
            client_id = yaml.safe_load(f)
        return client_id
    return {}


def get_hardware_usage(pid: int) -> t.Dict[str, t.Any]:
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
    uri: str = TRACKING_URI,
) -> None:
    if do_not_track():
        return
    payload = get_payload(
        event_type=event_type, event_pid=event_pid, event_properties=event_properties
    )
    send_usage_event(payload, timeout=timeout, uri=uri)


@experimental
def async_loop_forever(
    interval: int = get_usage_stats_interval_seconds(), cancellable: bool = False
) -> "t.Callable[[AsyncFunc[P]], AsyncFunc[P]]":
    def wrapper(coro: "AsyncFunc[P]") -> "AsyncFunc[P]":
        @wraps(coro)
        async def _(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
            while True:
                await asyncio.sleep(interval)
                try:
                    await coro(*args, **kwargs)
                except asyncio.CancelledError as err:
                    if cancellable:
                        logger.info(
                            f"An `async_loop_forever` coroutine is cancelled: {coro}."
                        )
                    else:
                        raise BentoMLException(
                            f"`async_loop_forever` {coro} cannot be cancelled since `cancellable={cancellable}`."
                        ) from err
                except Exception:  # pylint: disable=broad-except
                    logger.exception(f"Error looping coroutine {coro}.")

        return _

    return wrapper


async def track_async(
    event_type: str,
    event_pid: int,
    event_properties: t.Optional[t.Dict[str, t.Any]] = None,
    timeout: int = 2,
    uri: str = TRACKING_URI,
) -> None:
    """Async send the data. Using ThreadPool to implement async send."""
    # https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
    if do_not_track():
        return

    payload = get_payload(
        event_type=event_type, event_pid=event_pid, event_properties=event_properties
    )

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, send_usage_event, payload, timeout, uri)


@slient
@experimental
@async_loop_forever(get_usage_stats_interval_seconds())
async def scheduled_track(current_pid: int, event_properties: t.Dict[str, t.Any]):
    await track_async(
        BENTO_SERVE_SCHEDULED_TRACK_EVENT_TYPE,
        current_pid,
        event_properties=event_properties,
    )
