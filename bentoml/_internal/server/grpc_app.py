from __future__ import annotations

import typing as t
import logging
import functools
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

import psutil
from grpc import aio
from simple_di import inject
from simple_di import Provide

from bentoml.exceptions import MissingDependencyException

from .grpc.server import GRPCServer
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import grpc

    from ..service import Service

    OnStartup = list[t.Callable[[], None | t.Coroutine[t.Any, t.Any, None]]]
    Interceptors = list[t.Callable[[], aio.ServerInterceptor]]


class GRPCAppFactory:
    """
    GRPCApp creates an async gRPC API server based on APIs defined with a BentoService via BentoService#apis.
    This is a light wrapper around GRPCServer with addition to `on_startup` and `on_shutdown` hooks.

    Note that even though the code are similar with BaseAppFactory, gRPC protocol is different from ASGI.
    """

    @inject
    def __init__(
        self,
        bento_service: Service,
        *,
        maximum_concurrent_rpcs: int
        | None = Provide[BentoMLContainer.grpc.maximum_concurrent_rpcs],
        enable_metrics: bool = Provide[
            BentoMLContainer.api_server_config.metrics.enabled
        ],
    ) -> None:
        self.bento_service = bento_service
        self.enable_metrics = enable_metrics

        # Note that the max_workers are used inside ThreadPoolExecutor.
        # This ThreadPoolExecutor are used by aio.Server() to execute non-AsyncIO RPC handlers.
        # Setting it to 1 makes it thread-safe for sync APIs.
        self.max_workers = 1

        # maximum_concurrent_rpcs defines the maximum number of concurrent RPCs this server
        # will service before returning RESOURCE_EXHAUSTED status.
        # Set to None will indicate no limit.
        self.maximum_concurrent_rpcs = maximum_concurrent_rpcs

    @property
    def name(self) -> str:
        return self.bento_service.name

    @property
    def on_startup(self) -> OnStartup:
        on_startup: OnStartup = [self.bento_service.on_grpc_server_startup]
        if BentoMLContainer.development_mode.get():
            for runner in self.bento_service.runners:
                on_startup.append(functools.partial(runner.init_local, quiet=True))
        else:
            for runner in self.bento_service.runners:
                on_startup.append(runner.init_client)

        return on_startup

    @property
    def on_shutdown(self) -> list[t.Callable[[], None]]:
        on_shutdown = [self.bento_service.on_grpc_server_shutdown]
        for runner in self.bento_service.runners:
            on_shutdown.append(runner.destroy)

        return on_shutdown

    def __call__(self) -> GRPCServer:
        try:
            from grpc_health.v1 import health
        except ImportError:
            raise MissingDependencyException(
                "'grpcio-health-checking' is required for using health checking endpoints. Install with `pip install grpcio-health-checking`."
            )
        from .grpc.servicer import create_bento_servicer

        server = aio.server(
            migration_thread_pool=ThreadPoolExecutor(max_workers=self.max_workers),
            handlers=self.handlers,
            interceptors=self.interceptors,
            options=self.options,
            maximum_concurrent_rpcs=self.maximum_concurrent_rpcs,
        )

        # Create a health check servicer. We use the non-blocking implementation
        # to avoid thread starvation.
        health_servicer = health.aio.HealthServicer()

        return GRPCServer(
            server=server,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
            _health_servicer=health_servicer,
            _bento_servicer=create_bento_servicer(self.bento_service),
        )

    @property
    @inject
    def options(
        self,
        max_message_length: int
        | None = Provide[BentoMLContainer.grpc.max_message_length],
        max_concurrent_streams: int = Provide[
            BentoMLContainer.grpc.max_concurrent_streams
        ],
    ) -> aio.ChannelArgumentType:
        # TODO: refactor out a separate gRPC config class.
        options: list[tuple[str, t.Any]] = []
        if not psutil.WINDOWS:
            # https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/grpc_types.h#L294
            # Eventhough GRPC_ARG_ALLOW_REUSEPORT is set to 1 by default, we want still
            # want to explicitly set it to 1 so that we can spawn multiple gRPC servers in
            # production settings.
            options.append(("grpc.so_reuseport", 1))

        if max_concurrent_streams:
            # TODO: refactor max_concurrent_streams this to be configurable
            options.append(("grpc.max_concurrent_streams", max_concurrent_streams))

        if max_message_length:
            options.extend(
                (
                    ("grpc.max_message_length", max_message_length),
                    ("grpc.max_receive_message_length", max_message_length),
                    ("grpc.max_send_message_length", max_message_length),
                )
            )
        return tuple(options)

    @property
    def handlers(self) -> t.Sequence[grpc.GenericRpcHandler] | None:
        # Note that currently BentoML doesn't provide any specific
        # handlers for gRPC. If users have any specific handlers,
        # BentoML will pass it through to grpc.aio.Server
        if self.bento_service.grpc_handlers:
            return self.bento_service.grpc_handlers
        return

    @property
    def interceptors(self) -> list[aio.ServerInterceptor]:
        # Note that order of interceptors is important here.
        from .grpc.interceptors import GenericHeadersServerInterceptor
        from .grpc.interceptors.opentelemetry import AsyncOpenTelemetryServerInterceptor

        interceptors: Interceptors = [
            GenericHeadersServerInterceptor,
            AsyncOpenTelemetryServerInterceptor,
        ]

        if self.enable_metrics:
            from .grpc.interceptors.prometheus import PrometheusServerInterceptor

            interceptors.append(
                functools.partial(
                    PrometheusServerInterceptor,
                    bento_service=self.bento_service,
                )
            )

        if BentoMLContainer.api_server_config.logging.access.enabled.get():
            from .grpc.interceptors.access import AccessLogServerInterceptor

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                interceptors.append(AccessLogServerInterceptor)

        # add users-defined interceptors.
        interceptors.extend(self.bento_service.interceptors)

        return list(map(lambda x: x(), interceptors))
