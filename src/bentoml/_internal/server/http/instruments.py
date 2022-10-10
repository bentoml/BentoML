from __future__ import annotations

import logging
import contextvars
from timeit import default_timer
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ...context import component_context
from ...utils.metrics import metric_name
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ... import external_typing as ext
    from ...server.metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)
START_TIME_VAR: contextvars.ContextVar[float] = contextvars.ContextVar("START_TIME_VAR")
STATUS_VAR: contextvars.ContextVar[int] = contextvars.ContextVar("STATUS_VAR")


class HTTPTrafficMetricsMiddleware:
    def __init__(
        self,
        app: ext.ASGIApp,
        namespace: str = "bentoml_api_server",
    ):
        self.app = app
        self.namespace = namespace
        self._is_setup = False

    @inject
    def _setup(
        self,
        metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
        duration_buckets: tuple[float, ...] = Provide[
            BentoMLContainer.duration_buckets
        ],
    ):
        self.metrics_client = metrics_client

        DEFAULT_NAMESPACE = "bentoml_api_server"
        if self.namespace == DEFAULT_NAMESPACE:
            legacy_namespace = "BENTOML"
        else:
            legacy_namespace = self.namespace

        # legacy metrics names for bentoml<1.0.6
        self.legacy_metrics_request_duration = metrics_client.Histogram(
            namespace=legacy_namespace,
            name=metric_name(component_context.bento_name, "request_duration_seconds"),
            documentation="Legacy API HTTP request duration in seconds",
            labelnames=["endpoint", "service_version", "http_response_code"],
            buckets=duration_buckets,
        )
        self.legacy_metrics_request_total = metrics_client.Counter(
            namespace=legacy_namespace,
            name=metric_name(component_context.bento_name, "request_total"),
            documentation="Legacy total number of HTTP requests",
            labelnames=["endpoint", "service_version", "http_response_code"],
        )
        self.legacy_metrics_request_in_progress = metrics_client.Gauge(
            namespace=legacy_namespace,
            name=metric_name(component_context.bento_name, "request_in_progress"),
            documentation="Legacy total number of HTTP requests in progress now",
            labelnames=["endpoint", "service_version"],
            multiprocess_mode="livesum",
        )

        self.metrics_request_duration = metrics_client.Histogram(
            namespace=self.namespace,
            name="request_duration_seconds",
            documentation="API HTTP request duration in seconds",
            labelnames=[
                "endpoint",
                "service_name",
                "service_version",
                "http_response_code",
            ],
            buckets=duration_buckets,
        )
        self.metrics_request_total = metrics_client.Counter(
            namespace=self.namespace,
            name="request_total",
            documentation="Total number of HTTP requests",
            labelnames=[
                "endpoint",
                "service_name",
                "service_version",
                "http_response_code",
            ],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            namespace=self.namespace,
            name="request_in_progress",
            documentation="Total number of HTTP requests in progress now",
            labelnames=["endpoint", "service_name", "service_version"],
            multiprocess_mode="livesum",
        )
        self._is_setup = True

    async def __call__(
        self,
        scope: ext.ASGIScope,
        receive: ext.ASGIReceive,
        send: ext.ASGISend,
    ) -> None:
        if not self._is_setup:
            self._setup()
        if not scope["type"].startswith("http"):
            await self.app(scope, receive, send)
            return

        if scope["path"] == "/metrics":
            from starlette.responses import Response

            response = Response(
                self.metrics_client.generate_latest(),
                status_code=200,
                media_type=self.metrics_client.CONTENT_TYPE_LATEST,
            )
            await response(scope, receive, send)
            return

        endpoint = scope["path"]
        START_TIME_VAR.set(default_timer())

        async def wrapped_send(message: "ext.ASGIMessage") -> None:
            if message["type"] == "http.response.start":
                STATUS_VAR.set(message["status"])
            elif message["type"] == "http.response.body":
                if ("more_body" not in message) or not message["more_body"]:
                    assert START_TIME_VAR.get() != 0
                    assert STATUS_VAR.get() != 0

                    # instrument request total count
                    self.legacy_metrics_request_total.labels(
                        endpoint=endpoint,
                        service_version=component_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                    ).inc()
                    self.metrics_request_total.labels(
                        endpoint=endpoint,
                        service_name=component_context.bento_name,
                        service_version=component_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                    ).inc()

                    # instrument request duration
                    total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                    self.legacy_metrics_request_duration.labels(  # type: ignore
                        endpoint=endpoint,
                        service_version=component_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                    ).observe(total_time)
                    self.metrics_request_duration.labels(  # type: ignore
                        endpoint=endpoint,
                        service_name=component_context.bento_name,
                        service_version=component_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                    ).observe(total_time)

                    START_TIME_VAR.set(0)
                    STATUS_VAR.set(0)
            await send(message)

        with self.legacy_metrics_request_in_progress.labels(
            endpoint=endpoint, service_version=component_context.bento_version
        ).track_inprogress(), self.metrics_request_in_progress.labels(
            endpoint=endpoint,
            service_name=component_context.bento_name,
            service_version=component_context.bento_version,
        ).track_inprogress():
            await self.app(scope, receive, wrapped_send)
            return


class RunnerTrafficMetricsMiddleware:
    def __init__(
        self,
        app: "ext.ASGIApp",
        namespace: str = "bentoml_runner",
    ):
        self.app = app
        self.namespace = namespace
        self._is_setup = False

    @inject
    def _setup(
        self,
        metrics_client: "PrometheusClient" = Provide[BentoMLContainer.metrics_client],
    ):
        self.metrics_client = metrics_client

        self.metrics_request_duration = metrics_client.Histogram(
            namespace=self.namespace,
            name="request_duration_seconds",
            documentation="runner RPC duration in seconds",
            labelnames=[
                "endpoint",
                "service_name",
                "service_version",
                "http_response_code",
                "runner_name",
            ],
        )
        self.metrics_request_total = metrics_client.Counter(
            namespace=self.namespace,
            name="request_total",
            documentation="Total number of runner RPC",
            labelnames=[
                "endpoint",
                "service_name",
                "service_version",
                "http_response_code",
                "runner_name",
            ],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            namespace=self.namespace,
            name="request_in_progress",
            documentation="Total number of runner RPC in progress now",
            labelnames=["endpoint", "service_name", "service_version", "runner_name"],
            multiprocess_mode="livesum",
        )
        self._is_setup = True

    async def __call__(
        self,
        scope: "ext.ASGIScope",
        receive: "ext.ASGIReceive",
        send: "ext.ASGISend",
    ) -> None:
        if not self._is_setup:
            self._setup()
        if not scope["type"].startswith("http"):
            await self.app(scope, receive, send)
            return

        if scope["path"] == "/metrics":
            from starlette.responses import Response

            response = Response(
                self.metrics_client.generate_latest(),
                status_code=200,
                media_type=self.metrics_client.CONTENT_TYPE_LATEST,
            )
            await response(scope, receive, send)
            return

        endpoint = scope["path"]
        START_TIME_VAR.set(default_timer())

        async def wrapped_send(message: "ext.ASGIMessage") -> None:
            if message["type"] == "http.response.start":
                STATUS_VAR.set(message["status"])
            elif message["type"] == "http.response.body":
                if ("more_body" not in message) or not message["more_body"]:
                    assert START_TIME_VAR.get() != 0
                    assert STATUS_VAR.get() != 0

                    # instrument request total count
                    self.metrics_request_total.labels(
                        endpoint=endpoint,
                        service_name=component_context.bento_name,
                        service_version=component_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                        runner_name=component_context.component_name,
                    ).inc()

                    # instrument request duration
                    total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                    self.metrics_request_duration.labels(  # type: ignore
                        endpoint=endpoint,
                        service_name=component_context.bento_name,
                        service_version=component_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                        runner_name=component_context.component_name,
                    ).observe(total_time)

                    START_TIME_VAR.set(0)
                    STATUS_VAR.set(0)
            await send(message)

        with self.metrics_request_in_progress.labels(
            endpoint=endpoint,
            service_name=component_context.bento_name,
            service_version=component_context.bento_version,
            runner_name=component_context.component_name,
        ).track_inprogress():
            await self.app(scope, receive, wrapped_send)
            return
