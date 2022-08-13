from __future__ import annotations

import typing as t
import logging
import functools
import contextvars
from timeit import default_timer
from typing import TYPE_CHECKING

import grpc
from grpc import aio
from simple_di import inject
from simple_di import Provide

from bentoml.grpc.utils import to_http_status
from bentoml.grpc.utils import wrap_rpc_handler

from ....utils import LazyLoader
from ....configuration.containers import BentoMLContainer

START_TIME_VAR: contextvars.ContextVar[float] = contextvars.ContextVar("START_TIME_VAR")

if TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2
    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext

    from ....service import Service
    from ...metrics.prometheus import PrometheusClient
else:
    service_pb2 = LazyLoader("service_pb2", globals(), "bentoml.grpc.v1.service_pb2")


logger = logging.getLogger(__name__)


class PrometheusServerInterceptor(aio.ServerInterceptor):
    """
    An async interceptor for Prometheus metrics.
    """

    def __init__(self, bento_service: Service):
        self._is_setup = False
        self.bento_service = bento_service

    @inject
    def _setup(
        self,
        metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
    ):  # pylint: disable=attribute-defined-outside-init

        # a valid tag name may includes invalid characters, so we need to escape them
        # ref: https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
        service_name = self.bento_service.name.replace("-", ":").replace(".", "::")

        self.metrics_request_duration = metrics_client.Histogram(
            name=f"{service_name}_request_duration_seconds",
            documentation=f"{service_name} API GRPC request duration in seconds",
            labelnames=["api_name", "service_version", "http_response_code"],
        )
        self.metrics_request_total = metrics_client.Counter(
            name=f"{service_name}_request_total",
            documentation="Total number of GRPC requests",
            labelnames=["api_name", "service_version", "http_response_code"],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            name=f"{service_name}_request_in_progress",
            documentation="Total number of GRPC requests in progress now",
            labelnames=["api_name", "service_version"],
            multiprocess_mode="livesum",
        )
        self._is_setup = True

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        if not self._is_setup:
            self._setup()

        handler = await continuation(handler_call_details)

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        service_version = (
            self.bento_service.tag.version if self.bento_service.tag else ""
        )

        START_TIME_VAR.set(default_timer())

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behavior(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                if not isinstance(request, service_pb2.Request):
                    return await behaviour(request, context)

                api_name = request.api_name

                # instrument request total count
                self.metrics_request_total.labels(
                    api_name=api_name,
                    service_version=service_version,
                    http_response_code=to_http_status(
                        t.cast(grpc.StatusCode, context.code())
                    ),
                ).inc()

                # instrument request duration
                assert START_TIME_VAR.get() != 0
                total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                self.metrics_request_duration.labels(  # type: ignore (unfinished prometheus types)
                    api_name=api_name,
                    service_version=service_version,
                    http_response_code=to_http_status(
                        t.cast(grpc.StatusCode, context.code())
                    ),
                ).observe(
                    total_time
                )

                START_TIME_VAR.set(0)

                # instrument request in progress
                with self.metrics_request_in_progress.labels(
                    api_name=api_name, service_version=service_version
                ).track_inprogress():
                    response = behaviour(request, context)
                    if not hasattr(response, "__aiter__"):
                        response = await response
                return response

            return new_behavior

        return t.cast("RpcMethodHandler", wrap_rpc_handler(wrapper, handler))
