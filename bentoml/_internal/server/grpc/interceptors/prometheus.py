from __future__ import annotations

import typing as t
import logging
import functools
import contextvars
from timeit import default_timer
from typing import TYPE_CHECKING

from grpc import aio
from simple_di import inject
from simple_di import Provide

from ....utils import reserve_free_port
from ....utils.grpc import wrap_rpc_handler
from ....configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from bentoml._internal.server.grpc.types import Request
    from bentoml._internal.server.grpc.types import Response
    from bentoml._internal.server.grpc.types import RpcMethodHandler
    from bentoml._internal.server.grpc.types import AsyncHandlerMethod
    from bentoml._internal.server.grpc.types import HandlerCallDetails
    from bentoml._internal.server.grpc.types import BentoServicerContext

    from ....service import Service
    from ...metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)
START_TIME_VAR: "contextvars.ContextVar[float]" = contextvars.ContextVar(
    "START_TIME_VAR"
)


class PrometheusServerInterceptor(aio.ServerInterceptor):
    """
    An async interceptor for prometheus metrics
    """

    def __init__(
        self,
        bento_service: Service,
    ):
        self.bento_service = bento_service
        self._is_setup = False

    @inject
    def _setup(
        self,
        metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
    ):
        self.metrics_client = metrics_client
        service_name = self.bento_service.name
        # a valid tag name may includes invalid characters, so we need to escape them
        # ref: https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
        service_name = service_name.replace("-", ":").replace(".", "::")

        self.metrics_request_duration = metrics_client.Histogram(
            name=service_name + "_request_duration_seconds",
            documentation=service_name + " API GRPC request duration in seconds",
            labelnames=["api_name", "service_version", "grpc_response_code"],
        )
        self.metrics_request_total = metrics_client.Counter(
            name=service_name + "_request_total",
            documentation="Total number of GRPC requests",
            labelnames=["api_name", "service_version", "grpc_response_code"],
        )
        self.metrics_request_latency = metrics_client.Histogram(
            name=service_name + "_request_latency",
            documentation="API GRPC response latency",
            labelnames=["api_name", "service_version", "grpc_response_code"],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            name=service_name + "_request_in_progress",
            documentation="Total number of GRPC requests in progress now",
            labelnames=["api_name", "service_version"],
        )
        self._is_setup = True

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            if not self._is_setup:
                self._setup()
            service_version = (
                self.bento_service.tag.version
                if self.bento_service.tag is not None
                else ""
            )
            START_TIME_VAR.set(default_timer())

            @functools.wraps(behaviour)
            async def new_behavior(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                api_name = request.api_name

                # instrument request total count
                self.metrics_request_total.labels(
                    api_name=api_name,
                    service_version=service_version,
                    grpc_response_code=context.code(),
                ).inc()

                # instrument request duration
                assert START_TIME_VAR.get() != 0
                total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                self.metrics_request_duration.labels(  # type: ignore
                    api_name=api_name,
                    service_version=service_version,
                    grpc_response_code=context.code(),
                ).observe(total_time)
                START_TIME_VAR.set(0)

                # instrument latency
                self.metrics_request_latency.labels(  # type: ignore
                    api_name=api_name,
                    service_version=service_version,
                    grpc_response_code=context.code(),
                ).observe(max(default_timer() - START_TIME_VAR.get(), 0))

                # instrument request in progress
                with self.metrics_request_in_progress.labels(
                    api_name=api_name, service_version=service_version
                ).track_inprogress():
                    response = behaviour(request, context)
                    if not hasattr(response, "__aiter__"):
                        response = await response
                return response

            with reserve_free_port() as port:
                logger.info(
                    f"Prometheus metrics for grpc server can be viewed at http://127.0.0.1:{port}/"
                )
                self.metrics_client.start_http_server(port)
            return new_behavior

        return wrap_rpc_handler(wrapper, handler)
