from __future__ import annotations

import typing as t
import logging
import functools
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from grpc import aio
from simple_di import inject
from simple_di import Provide

from bentoml.exceptions import MissingDependencyException

from .grpc.server import GRPCServer
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..service import Service

    OnStartup = list[t.Callable[[], None | t.Coroutine[t.Any, t.Any, None]]]


class GRPCAppFactory:
    """
    GRPCApp creates an async gRPC API server based on APIs defined with a BentoService via BentoService#apis.
    This is a light wrapper around GRPCServer with addition to `on_startup` and `on_shutdown` hooks.

    Note that even though the code are similar with BaseAppFactory, gRPC protocol is different from ASGI.
    """

    _is_ready: bool = False

    @inject
    def __init__(
        self,
        bento_service: Service,
        *,
        _thread_pool_size: int = 10,
        maximum_concurrent_rpcs: int
        | None = Provide[BentoMLContainer.grpc.maximum_concurrent_rpcs],
    ) -> None:
        self.bento_service = bento_service
        self.server = aio.server(
            ThreadPoolExecutor(_thread_pool_size),
            interceptors=self.interceptors,
            options=self.options,
            maximum_concurrent_rpcs=maximum_concurrent_rpcs,
        )

    @property
    def name(self) -> str:
        return self.bento_service.name

    def mark_as_ready(self) -> None:
        self._is_ready = True

    @property
    def on_startup(self) -> OnStartup:
        on_startup: OnStartup = [
            self.mark_as_ready,
            self.bento_service.on_grpc_server_startup,
        ]
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

        # Create a health check servicer. We use the non-blocking implementation
        # to avoid thread starvation.
        health_servicer = health.aio.HealthServicer()
        bento_servicer = create_bento_servicer(self.bento_service)

        return GRPCServer(
            server=self.server,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
            _health_servicer=health_servicer,
            _bento_servicer=bento_servicer,
        )

    @property
    @inject
    def options(
        self,
        max_message_length: int
        | None = Provide[BentoMLContainer.grpc.max_message_length],
    ) -> list[tuple[str, t.Any]]:
        options: list[tuple[str, t.Any]] = []
        if max_message_length:
            options.extend(
                [
                    ("grpc.max_message_length", max_message_length),
                    ("grpc.max_receive_message_length", max_message_length),
                    ("grpc.max_send_message_length", max_message_length),
                ]
            )
        return options

    @property
    def interceptors(self) -> list[aio.ServerInterceptor]:
        # Note that order of interceptors is important here.
        from .grpc.interceptors import GenericHeadersServerInterceptor

        # TODO: prometheus interceptors.
        interceptors: list[t.Type[aio.ServerInterceptor]] = [
            GenericHeadersServerInterceptor,
        ]

        access_log_config = BentoMLContainer.api_server_config.logging.access
        if access_log_config.enabled.get():
            from .grpc.interceptors import AccessLogServerInterceptor

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                interceptors.append(AccessLogServerInterceptor)

        # add users-defined interceptors.
        interceptors.extend(self.bento_service.interceptors)

        return list(map(lambda x: x(), interceptors))
