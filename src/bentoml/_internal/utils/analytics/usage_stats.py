from __future__ import annotations

import contextlib
import logging
import os
import secrets
import threading
import typing as t
from datetime import datetime
from datetime import timezone
from functools import lru_cache
from functools import wraps
from typing import TYPE_CHECKING

import attr
import httpx
from simple_di import Provide
from simple_di import inject

from ...configuration import get_debug_mode
from ...configuration.containers import BentoMLContainer
from ...utils import compose
from .schemas import CommonProperties
from .schemas import EventMeta
from .schemas import ServeInitEvent
from .schemas import ServeUpdateEvent
from .schemas import TrackingPayload

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    T = t.TypeVar("T")
    AsyncFunc = t.Callable[P, t.Coroutine[t.Any, t.Any, t.Any]]

    from _bentoml_sdk import Service as NewService
    from prometheus_client.samples import Sample

    from bentoml import Service

    from ...server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

BENTOML_DO_NOT_TRACK = "BENTOML_DO_NOT_TRACK"
BENTOML_SERVE_FROM_SERVER_API = "__BENTOML_SERVE_FROM_SERVER_API"
USAGE_TRACKING_URL = "https://t.bentoml.com"
SERVE_USAGE_TRACKING_INTERVAL_SECONDS = int(12 * 60 * 60)  # every 12 hours
USAGE_REQUEST_TIMEOUT_SECONDS = 1


@lru_cache(maxsize=None)
def _bentoml_serve_from_server_api() -> bool:
    return os.environ.get(BENTOML_SERVE_FROM_SERVER_API, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def do_not_track() -> bool:  # pragma: no cover
    # Returns True if and only if the environment variable is defined and has value True.
    # The function is cached for better performance.
    return os.environ.get(BENTOML_DO_NOT_TRACK, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
    # For BentoML developers only - debug and print event payload if turned on
    return os.environ.get("__BENTOML_DEBUG_USAGE", str(False)).lower() == "true"


def silent(func: t.Callable[P, T]) -> t.Callable[P, T]:  # pragma: no cover
    # Silent errors when tracking
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            if _usage_event_debugging():
                if get_debug_mode():
                    logger.error(
                        "Tracking Error: %s", err, stack_info=True, stacklevel=3
                    )
                else:
                    logger.info("Tracking Error: %s", err)
            else:
                logger.debug("Tracking Error: %s", err)

    return wrapper


@attr.define
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


@silent
def track(event_properties: EventMeta):
    if do_not_track():
        return
    payload = get_payload(event_properties=event_properties)

    if _usage_event_debugging():
        # For internal debugging purpose
        logger.info("Tracking Payload: %s", payload)
        return

    httpx.post(USAGE_TRACKING_URL, json=payload, timeout=USAGE_REQUEST_TIMEOUT_SECONDS)


@inject
def _track_serve_init(
    svc: Service | NewService[t.Any],
    production: bool,
    serve_kind: str,
    from_server_api: bool,
    serve_info: ServeInfo = Provide[BentoMLContainer.serve_info],
):
    from bentoml import Service

    is_legacy = isinstance(svc, Service)

    if svc.bento is not None:
        bento = svc.bento
        event_properties = ServeInitEvent(
            serve_id=serve_info.serve_id,
            serve_from_bento=True,
            serve_from_server_api=from_server_api,
            production=production,
            serve_kind=serve_kind,
            bento_creation_timestamp=bento.info.creation_time,
            num_of_models=len(bento.info.models),
            num_of_runners=len(svc.runners) if is_legacy else len(svc.dependencies),
            num_of_apis=len(bento.info.apis),
            model_types=[m.module for m in bento.info.models],
            runnable_types=[r.runnable_type for r in bento.info.runners],
            api_input_types=[api.input_type for api in bento.info.apis],
            api_output_types=[api.output_type for api in bento.info.apis],
        )
    else:
        if is_legacy:
            num_models = len(
                set(
                    svc.models
                    + [model for runner in svc.runners for model in runner.models]
                )
            )
        else:
            from bentoml import Model

            def _get_models(svc: NewService[t.Any], seen: set[str]) -> t.Set[Model]:
                if svc.name in seen:
                    return set()
                seen.add(svc.name)
                models = set(svc.models)
                for dep in svc.dependencies.values():
                    models.update(_get_models(dep.on, seen))
                return models

            num_models = len(_get_models(svc, set()))

        event_properties = ServeInitEvent(
            serve_id=serve_info.serve_id,
            serve_from_bento=False,
            serve_from_server_api=from_server_api,
            production=production,
            serve_kind=serve_kind,
            bento_creation_timestamp=None,
            num_of_models=num_models,
            num_of_runners=len(svc.runners) if is_legacy else len(svc.dependencies),
            num_of_apis=len(svc.apis.keys()),
            runnable_types=[r.runnable_class.__name__ for r in svc.runners]
            if is_legacy
            else [d.on.name for d in svc.dependencies.values()],
            api_input_types=[api.input.__class__.__name__ for api in svc.apis.values()]
            if is_legacy
            else [],
            api_output_types=[
                api.output.__class__.__name__ for api in svc.apis.values()
            ]
            if is_legacy
            else [],
        )

    track(event_properties)


EXCLUDE_PATHS = {"/docs.json", "/livez", "/healthz", "/readyz"}


def filter_metrics(
    samples: list[Sample], *filters: t.Callable[[list[Sample]], list[Sample]]
):
    return [
        {**sample.labels, "value": sample.value}
        for sample in compose(*filters)(samples)
    ]


def get_metrics_report(
    metrics_client: PrometheusClient,
    serve_kind: str,
) -> list[dict[str, str | float]]:
    """
    Get Prometheus metrics reports from the metrics client. This will be used to determine tracking events.
    If the return metrics are legacy metrics, the metrics will have prefix BENTOML_, otherwise they will have prefix bentoml_

    Args:
        metrics_client: Instance of bentoml._internal.server.metrics.prometheus.PrometheusClient
        grpc: Whether the metrics are for gRPC server.

    Returns:
        A tuple of a list of metrics and an optional boolean to determine whether the return metrics are legacy metrics.
    """
    for metric in metrics_client.text_string_to_metric_families():
        metric_type = t.cast("str", metric.type)  # type: ignore (we need to cast due to no prometheus types)
        metric_name = t.cast("str", metric.name)  # type: ignore (we need to cast due to no prometheus types)
        metric_samples = t.cast("list[Sample]", metric.samples)  # type: ignore (we need to cast due to no prometheus types)
        if metric_type != "counter":
            continue
        # We only care about the counter metrics.
        assert metric_type == "counter"
        if serve_kind == "grpc":
            _filters: list[t.Callable[[list[Sample]], list[Sample]]] = [
                lambda samples: [s for s in samples if "api_name" in s.labels]
            ]
        elif serve_kind == "http":
            _filters = [
                lambda samples: [
                    s
                    for s in samples
                    if not s.labels["endpoint"].startswith("/static_content/")
                ],
                lambda samples: [
                    s for s in samples if s.labels["endpoint"] not in EXCLUDE_PATHS
                ],
                lambda samples: [s for s in samples if "endpoint" in s.labels],
            ]
        else:
            raise NotImplementedError("Unknown serve kind %s" % serve_kind)
        # If metrics prefix is BENTOML_, this is legacy metrics
        if metric_name.endswith("_request") and (
            metric_name.startswith("bentoml_") or metric_name.startswith("BENTOML_")
        ):
            return filter_metrics(metric_samples, *_filters)

    return []


@inject
@contextlib.contextmanager
def track_serve(
    svc: Service | NewService[t.Any],
    *,
    production: bool = False,
    from_server_api: bool | None = None,
    serve_kind: str = "http",
    component: str = "standalone",
    metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
    serve_info: ServeInfo = Provide[BentoMLContainer.serve_info],
) -> t.Generator[None, None, None]:
    if do_not_track():
        yield
        return

    if from_server_api is None:
        from_server_api = _bentoml_serve_from_server_api()

    _track_serve_init(
        svc=svc,
        production=production,
        serve_kind=serve_kind,
        from_server_api=from_server_api,
    )

    if _usage_event_debugging():
        tracking_interval = 5
    else:
        tracking_interval = SERVE_USAGE_TRACKING_INTERVAL_SECONDS  # pragma: no cover

    stop_event = threading.Event()

    @silent
    def loop() -> t.NoReturn:  # type: ignore
        last_tracked_timestamp: datetime = serve_info.serve_started_timestamp
        while not stop_event.wait(tracking_interval):  # pragma: no cover
            now = datetime.now(timezone.utc)
            event_properties = ServeUpdateEvent(
                serve_id=serve_info.serve_id,
                production=production,
                # Note that we are currently only have two tracking jobs: http and grpc
                serve_kind=serve_kind,
                # Current accept components are "standalone", "api_server" and "runner"
                component=component,
                # check if serve is running from server API or just normal CLI
                serve_from_server_api=from_server_api,
                triggered_at=now,
                duration_in_seconds=int((now - last_tracked_timestamp).total_seconds()),
                metrics=get_metrics_report(metrics_client, serve_kind=serve_kind),
            )
            last_tracked_timestamp = now
            track(event_properties)

    tracking_thread = threading.Thread(target=loop, daemon=True)
    try:
        tracking_thread.start()
        yield
    finally:
        stop_event.set()
        tracking_thread.join()
