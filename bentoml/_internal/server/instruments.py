import logging
import contextvars
from timeit import default_timer
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ..configuration.containers import DeploymentContainer

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..service import Service
    from ..server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)
START_TIME_VAR: "contextvars.ContextVar[float]" = contextvars.ContextVar(
    "START_TIME_VAR"
)


class MetricsMiddleware:
    @inject
    def __init__(
        self,
        app: "ext.ASGIApp",
        bento_service: "Service",
        metrics_client: "PrometheusClient" = Provide[
            DeploymentContainer.metrics_client
        ],
    ):
        self.app = app
        self.bento_service = bento_service

        service_name = self.bento_service.name

        self.metrics_request_duration = metrics_client.Histogram(
            name=service_name + "_request_duration_seconds",
            documentation=service_name + " API HTTP request duration in seconds",
            labelnames=["endpoint", "service_version", "http_response_code"],
        )
        self.metrics_request_total = metrics_client.Counter(
            name=service_name + "_request_total",
            documentation="Total number of HTTP requests",
            labelnames=["endpoint", "service_version", "http_response_code"],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            name=service_name + "_request_in_progress",
            documentation="Total number of HTTP requests in progress now",
            labelnames=["endpoint", "service_version"],
            multiprocess_mode="livesum",
        )

    @inject
    async def __call__(
        self,
        scope: "ext.ASGIScope",
        receive: "ext.ASGIReceive",
        send: "ext.ASGISend",
        metrics_client: "PrometheusClient" = Provide[
            DeploymentContainer.metrics_client
        ],
    ) -> None:
        if not scope["type"].startswith("http"):
            await self.app(scope, receive, send)
            return

        if scope["path"] == "/metrics":
            from starlette.responses import Response

            response = Response(
                metrics_client.generate_latest(),
                status_code=200,
                media_type=metrics_client.CONTENT_TYPE_LATEST,
            )
            await response(scope, receive, send)
            return

        service_version = (
            self.bento_service.version if self.bento_service.version is not None else ""
        )
        endpoint = scope["path"]
        START_TIME_VAR.set(default_timer())

        async def wrapped_send(message: "ext.ASGIMessage") -> None:
            if message["type"] == "http.response.start":
                status_code = message["status"]

                # instrument request total count
                self.metrics_request_total.labels(
                    endpoint=endpoint,
                    service_version=service_version,
                    http_response_code=status_code,
                ).inc()

                # instrument request duration
                assert START_TIME_VAR.get() != 0
                total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                self.metrics_request_duration.labels(  # type: ignore
                    endpoint=endpoint,
                    service_version=service_version,
                    http_response_code=status_code,
                ).observe(total_time)
            START_TIME_VAR.set(0)
            await send(message)

        with self.metrics_request_in_progress.labels(
            endpoint=endpoint, service_version=service_version
        ).track_inprogress():
            await self.app(scope, receive, wrapped_send)
            return
