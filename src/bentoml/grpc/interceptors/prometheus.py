from __future__ import annotations

import typing as t
import logging
import functools
import contextvars
from timeit import default_timer
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import to_http_status
from bentoml.grpc.utils import wrap_rpc_handler
from bentoml.grpc.utils import import_generated_stubs
from bentoml._internal.context import component_context
from bentoml._internal.configuration.containers import BentoMLContainer

START_TIME_VAR: contextvars.ContextVar[float] = contextvars.ContextVar("START_TIME_VAR")

if TYPE_CHECKING:
    import grpc
    from grpc import aio

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.types import Request
    from bentoml.grpc.types import Response
    from bentoml.grpc.types import RpcMethodHandler
    from bentoml.grpc.types import AsyncHandlerMethod
    from bentoml.grpc.types import HandlerCallDetails
    from bentoml.grpc.types import BentoServicerContext
    from bentoml._internal.server.metrics.prometheus import PrometheusClient
else:
    pb, _ = import_generated_stubs()
    grpc, aio = import_grpc()


logger = logging.getLogger(__name__)


class PrometheusServerInterceptor(aio.ServerInterceptor):
    """
    An async interceptor for Prometheus metrics.
    """

    def __init__(self, *, namespace: str = "bentoml_api_server"):
        self._is_setup = False
        self.namespace = namespace

    @inject
    def _setup(
        self,
        metrics_client: PrometheusClient = Provide[BentoMLContainer.metrics_client],
        duration_buckets: tuple[float, ...] = Provide[
            BentoMLContainer.duration_buckets
        ],
    ):  # pylint: disable=attribute-defined-outside-init
        self.metrics_request_duration = metrics_client.Histogram(
            namespace=self.namespace,
            name="request_duration_seconds",
            documentation="API GRPC request duration in seconds",
            labelnames=[
                "api_name",
                "service_name",
                "service_version",
                "http_response_code",
            ],
            buckets=duration_buckets,
        )
        self.metrics_request_total = metrics_client.Counter(
            namespace=self.namespace,
            name="request_total",
            documentation="Total number of GRPC requests",
            labelnames=[
                "api_name",
                "service_name",
                "service_version",
                "http_response_code",
            ],
        )
        self.metrics_request_in_progress = metrics_client.Gauge(
            namespace=self.namespace,
            name="request_in_progress",
            documentation="Total number of GRPC requests in progress now",
            labelnames=["api_name", "service_name", "service_version"],
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

        START_TIME_VAR.set(default_timer())

        def wrapper(behaviour: AsyncHandlerMethod[Response]):
            @functools.wraps(behaviour)
            async def new_behaviour(
                request: Request, context: BentoServicerContext
            ) -> Response | t.Awaitable[Response]:
                if not isinstance(request, pb.Request):
                    return await behaviour(request, context)

                api_name = request.api_name

                # instrument request total count
                self.metrics_request_total.labels(
                    api_name=api_name,
                    service_name=component_context.bento_name,
                    service_version=component_context.bento_version,
                    http_response_code=to_http_status(
                        t.cast(grpc.StatusCode, context.code())
                    ),
                ).inc()

                # instrument request duration
                assert START_TIME_VAR.get() != 0
                total_time = max(default_timer() - START_TIME_VAR.get(), 0)
                self.metrics_request_duration.labels(  # type: ignore (unfinished prometheus types)
                    api_name=api_name,
                    service_name=component_context.bento_name,
                    service_version=component_context.bento_version,
                    http_response_code=to_http_status(
                        t.cast(grpc.StatusCode, context.code())
                    ),
                ).observe(
                    total_time
                )
                START_TIME_VAR.set(0)
                # instrument request in progress
                with self.metrics_request_in_progress.labels(
                    api_name=api_name,
                    service_version=component_context.bento_version,
                    service_name=component_context.bento_name,
                ).track_inprogress():
                    response = await behaviour(request, context)
                return response

            return new_behaviour

        return wrap_rpc_handler(wrapper, handler)
