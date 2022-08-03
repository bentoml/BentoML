from __future__ import annotations

import typing as t
import logging
import functools
from timeit import default_timer
from typing import TYPE_CHECKING

from grpc import aio

from ....utils.grpc import to_http_status
from ....utils.grpc import wrap_rpc_handler

if TYPE_CHECKING:
    from ..types import Request
    from ..types import Response
    from ..types import RpcMethodHandler
    from ..types import AsyncHandlerMethod
    from ..types import HandlerCallDetails
    from ..types import BentoServicerContext
    from ....service import Service
    from ...metrics.prometheus import PrometheusClient

logger = logging.getLogger(__name__)


class PrometheusServerInterceptor(aio.ServerInterceptor):
    """
    An async interceptor for prometheus metrics
    """

    def __init__(
        self,
        bento_service: Service,
        metrics_client: PrometheusClient,
    ):
        self.metrics_client = metrics_client

        self.service_name = bento_service.name
        # a valid tag name may includes invalid characters, so we need to escape them
        # ref: https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
        self.service_name = self.service_name.replace("-", ":").replace(".", "::")

        self.service_version = (
            bento_service.tag.version if bento_service.tag is not None else ""
        )

        self.metrics_request_duration = metrics_client.Histogram(
            name=f"{self.service_name}_request_duration_seconds",
            documentation=f"{self.service_name} API GRPC request duration in seconds",
            labelnames=["api_name", "service_version", "http_response_code"],
        )
        self.metrics_request_total = metrics_client.Counter(
            name=f"{self.service_name}_request_total",
            documentation="Total number of GRPC requests",
            labelnames=["api_name", "service_version", "http_response_code"],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            name=f"{self.service_name}_request_in_progress",
            documentation="Total number of GRPC requests in progress now",
            labelnames=["api_name", "service_version"],
            multiprocess_mode="livesum",
        )

    async def intercept_service(
        self,
        continuation: t.Callable[[HandlerCallDetails], t.Awaitable[RpcMethodHandler]],
        handler_call_details: HandlerCallDetails,
    ) -> RpcMethodHandler:
        handler = await continuation(handler_call_details)

        if handler and (handler.response_streaming or handler.request_streaming):
            return handler

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behavior(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                api_name = request.api_name

                # instrument request total count
                self.metrics_request_total.labels(
                    api_name=api_name,
                    service_version=self.service_version,
                    http_response_code=to_http_status(context.code()),
                ).inc()

                # instrument request duration
                start = default_timer()
                self.metrics_request_duration.labels(  # type: ignore (unfinished prometheus types)
                    api_name=api_name,
                    service_version=self.service_version,
                    http_response_code=to_http_status(context.code()),
                ).observe(
                    max(default_timer() - start, 0)
                )
                start = 0

                # instrument request in progress
                with self.metrics_request_in_progress.labels(
                    api_name=api_name, service_version=self.service_version
                ).track_inprogress():
                    response = behaviour(request, context)
                    if not hasattr(response, "__aiter__"):
                        response = await response
                return response

            return new_behavior

        return wrap_rpc_handler(wrapper, handler)
