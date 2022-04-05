import os
import typing as t
import logging
import secrets
import threading
import contextlib
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from functools import wraps
from functools import lru_cache

import attrs
import requests
from simple_di import inject
from simple_di import Provide

from .schemas import EventMeta
from .schemas import ServeInitEvent
from .schemas import TrackingPayload
from .schemas import CommonProperties
from .schemas import ServeUpdateEvent
from ...configuration.containers import BentoMLContainer
from ...configuration.containers import DeploymentContainer

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

    from bentoml import Service

    from ...server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

BENTOML_DO_NOT_TRACK = "BENTOML_DO_NOT_TRACK"
USAGE_TRACKING_URL = "https://t.bentoml.com"
SERVE_USAGE_TRACKING_INTERVAL_SECONDS = int(12 * 60 * 60)  # every 12 hours
USAGE_REQUEST_TIMEOUT_SECONDS = 1


@lru_cache(maxsize=1)
def do_not_track() -> bool:
    # Returns True if and only if the environment variable is defined and has value True.
    # The function is cached for better performance.
    return os.environ.get(BENTOML_DO_NOT_TRACK, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
    # For BentoML developers only - debug and print event payload if turned on
    return os.environ.get("__BENTOML_DEBUG_USAGE", str(False)).lower() == "true"


def slient(func: "t.Callable[P, T]") -> "t.Callable[P, T]":  # pragma: no cover
    # Slient errors when tracking
    @wraps(func)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            if _usage_event_debugging():
                logger.info(f"Tracking Error: {err}")
            else:
                logger.debug(f"Tracking Error: {err}")

    return wrapper


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
) -> t.Dict[str, t.Any]:
    return TrackingPayload(
        session_id=session_id,
        common_properties=CommonProperties(),
        event_properties=event_properties,
        event_type=event_properties.event_name,
    ).to_dict()


@slient
def track(
    event_properties: EventMeta,
):
    if do_not_track():
        return
    payload = get_payload(event_properties=event_properties)

    if _usage_event_debugging():
        # For internal debugging purpose
        global SERVE_USAGE_TRACKING_INTERVAL_SECONDS  # pylint: disable=global-statement
        SERVE_USAGE_TRACKING_INTERVAL_SECONDS = 5
        logger.info("Tracking Payload: %s", payload)
        return

    requests.post(
        USAGE_TRACKING_URL, json=payload, timeout=USAGE_REQUEST_TIMEOUT_SECONDS
    )


@inject
def _track_serve_init(
    svc: "t.Optional[Service]",
    production: bool,
    serve_info: ServeInfo = Provide[DeploymentContainer.serve_info],
):
    if svc.bento is not None:
        bento = svc.bento
        event_properties = ServeInitEvent(
            serve_id=serve_info.serve_id,
            serve_from_bento=True,
            production=production,
            bento_creation_timestamp=bento.info.creation_time,
            num_of_models=len(bento.info.models),
            num_of_runners=len(svc.runners),
            num_of_apis=len(bento.info.apis),
            model_types=[m.module for m in bento.info.models],
            runner_types=[r.runner_type for r in bento.info.runners],
            api_input_types=[api.input_type for api in bento.info.apis],
            api_output_types=[api.output_type for api in bento.info.apis],
        )
    else:
        from ...frameworks.common.model_runner import BaseModelRunner
        from ...frameworks.common.model_runner import BaseModelSimpleRunner

        event_properties = ServeInitEvent(
            serve_id=serve_info.serve_id,
            serve_from_bento=False,
            production=production,
            bento_creation_timestamp=None,
            num_of_models=len(
                [
                    r
                    for r in svc.runners
                    if isinstance(r, (BaseModelRunner, BaseModelSimpleRunner))
                ]
            ),
            num_of_runners=len(svc.runners),
            num_of_apis=len(svc.apis.keys()),
            runner_types=[type(v).__name__ for v in svc.runners.values()],
            api_input_types=[api.input.__class__.__name__ for api in svc.apis.values()],
            api_output_types=[
                api.output.__class__.__name__ for api in svc.apis.values()
            ],
        )

    track(event_properties)


@inject
@contextlib.contextmanager
def track_serve(
    svc: "t.Optional[Service]",
    production: bool,
    metrics_client: "PrometheusClient" = Provide[DeploymentContainer.metrics_client],
    serve_info: ServeInfo = Provide[DeploymentContainer.serve_info],
):  # pragma: no cover
    if do_not_track():
        yield
        return

    _track_serve_init(svc, production)

    stop_event = threading.Event()

    @slient
    def loop() -> t.NoReturn:  # type: ignore
        while not stop_event.wait(SERVE_USAGE_TRACKING_INTERVAL_SECONDS):
            now = datetime.now(timezone.utc)
            event_properties = ServeUpdateEvent(
                serve_id=serve_info.serve_id,
                production=production,
                triggered_at=now,
                duration_in_seconds=(now - serve_info.serve_started_timestamp).seconds,
                metrics=metrics_client.get_metrics_report(),
            )
            track(event_properties)

    tracking_thread = threading.Thread(target=loop, daemon=True)
    try:
        tracking_thread.start()
        yield
    finally:
        stop_event.set()
        tracking_thread.join()
