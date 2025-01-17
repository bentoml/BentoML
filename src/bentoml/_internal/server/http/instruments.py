from __future__ import annotations

import contextvars
import logging
from timeit import default_timer
from typing import TYPE_CHECKING

from simple_di import Provide
from simple_di import inject

from ...configuration.containers import BentoMLContainer
from ...context import server_context

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
                    self.metrics_request_total.labels(
                        endpoint=endpoint,
                        service_name=server_context.bento_name,
                        service_version=server_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                    ).inc()

                    # instrument request duration
                    total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                    self.metrics_request_duration.labels(  # type: ignore
                        endpoint=endpoint,
                        service_name=server_context.bento_name,
                        service_version=server_context.bento_version,
                        http_response_code=STATUS_VAR.get(),
                    ).observe(total_time)

                    START_TIME_VAR.set(0)
                    STATUS_VAR.set(0)
            await send(message)

        with self.metrics_request_in_progress.labels(
            endpoint=endpoint,
            service_name=server_context.bento_name,
            service_version=server_context.bento_version,
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
        duration_buckets: tuple[float, ...] = Provide[
            BentoMLContainer.duration_buckets
        ],
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
            buckets=duration_buckets,
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
        self.metrics_websocket_connections = metrics_client.Gauge(
            namespace=self.namespace,
            name="websocket_connections",
            documentation="Total number of websocket connections",
            labelnames=["endpoint", "service_name", "service_version", "runner_name"],
            multiprocess_mode="livesum",
        )
        self.metrics_websocket_data_received = metrics_client.Summary(
            namespace=self.namespace,
            name="websocket_data_received",
            documentation="Total number of bytes received from websocket",
            labelnames=["endpoint", "service_name", "service_version", "runner_name"],
        )
        self.metrics_websocket_data_sent = metrics_client.Summary(
            namespace=self.namespace,
            name="websocket_data_sent",
            documentation="Total number of bytes sent to websocket",
            labelnames=["endpoint", "service_name", "service_version", "runner_name"],
        )
        # place holder metrics to initialize endpoints with 0 to facilitate rate calculation
        # otherwise, rate() can be 0 if the endpoint is hit for the first time
        for endpoint in server_context.service_routes:
            self.metrics_request_total.labels(
                endpoint=endpoint,
                service_name=server_context.bento_name,
                service_version=server_context.bento_version,
                http_response_code=200,
                runner_name=server_context.service_name,
            )
            self.metrics_request_total.labels(
                endpoint=endpoint,
                service_name=server_context.bento_name,
                service_version=server_context.bento_version,
                http_response_code=500,
                runner_name=server_context.service_name,
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
        if not scope["type"].startswith(("http", "websocket")):
            await self.app(scope, receive, send)
            return

        if scope["type"].startswith("http") and scope["path"] == "/metrics":
            from starlette.responses import Response

            response = Response(
                self.metrics_client.generate_latest(),
                status_code=200,
                media_type=self.metrics_client.CONTENT_TYPE_LATEST,
            )
            await response(scope, receive, send)
            return

        endpoint = scope["path"]
        start_time = default_timer()
        status_code = 0

        async def wrapped_receive() -> "ext.ASGIMessage":
            message = await receive()
            if message["type"] == "websocket.disconnect":
                self.metrics_websocket_connections.labels(
                    endpoint=endpoint,
                    service_name=server_context.bento_name,
                    service_version=server_context.bento_version,
                    runner_name=server_context.service_name,
                ).dec()
            elif message["type"] == "websocket.receive":
                if message.get("bytes") is not None:
                    data_len = len(message["bytes"])
                else:
                    data_len = len(message["text"])
                self.metrics_websocket_data_received.labels(
                    endpoint=endpoint,
                    service_name=server_context.bento_name,
                    service_version=server_context.bento_version,
                    runner_name=server_context.service_name,
                ).observe(data_len)
            return message

        async def wrapped_send(message: "ext.ASGIMessage") -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                if ("more_body" not in message) or not message["more_body"]:
                    # instrument request total count
                    self.metrics_request_total.labels(
                        endpoint=endpoint,
                        service_name=server_context.bento_name,
                        service_version=server_context.bento_version,
                        http_response_code=status_code,
                        runner_name=server_context.service_name,
                    ).inc()

                    # instrument request duration
                    total_time = max(default_timer() - start_time, 0)
                    self.metrics_request_duration.labels(  # type: ignore
                        endpoint=endpoint,
                        service_name=server_context.bento_name,
                        service_version=server_context.bento_version,
                        http_response_code=status_code,
                        runner_name=server_context.service_name,
                    ).observe(total_time)
            elif message["type"] == "websocket.send":
                if message.get("bytes") is not None:
                    data_len = len(message["bytes"])
                else:
                    data_len = len(message["text"])
                self.metrics_websocket_data_sent.labels(
                    endpoint=endpoint,
                    service_name=server_context.bento_name,
                    service_version=server_context.bento_version,
                    runner_name=server_context.service_name,
                ).observe(data_len)
            elif message["type"] == "websocket.accept":
                self.metrics_websocket_connections.labels(
                    endpoint=endpoint,
                    service_name=server_context.bento_name,
                    service_version=server_context.bento_version,
                    runner_name=server_context.service_name,
                ).inc()
            elif message["type"] == "websocket.close":
                self.metrics_websocket_connections.labels(
                    endpoint=endpoint,
                    service_name=server_context.bento_name,
                    service_version=server_context.bento_version,
                    runner_name=server_context.service_name,
                ).dec()

            await send(message)

        with self.metrics_request_in_progress.labels(
            endpoint=endpoint,
            service_name=server_context.bento_name,
            service_version=server_context.bento_version,
            runner_name=server_context.service_name,
        ).track_inprogress():
            await self.app(scope, wrapped_receive, wrapped_send)
            return
