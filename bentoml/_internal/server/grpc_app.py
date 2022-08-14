from __future__ import annotations

import typing as t
import logging
import functools
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from grpc import aio
from simple_di import inject
from simple_di import Provide

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

    def __init__(self, bento_service: Service, *, _thread_pool_size: int = 10) -> None:
        self.bento_service = bento_service
        self.server = aio.server(
            ThreadPoolExecutor(_thread_pool_size),
            interceptors=self.interceptors,
            options=self.options,
        )

    @property
    def name(self) -> str:
        return self.bento_service.name

    def mark_as_ready(self) -> None:
        self._is_ready = True

    @property
    def on_startup(self) -> OnStartup:
        from .grpc import register_bento_servicer
        from .grpc import register_health_servicer

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

        on_startup.extend(
            [
                functools.partial(
                    register_bento_servicer,
                    service=self.bento_service,
                    server=self.server,
                ),
                functools.partial(register_health_servicer, server=self.server),
            ]
        )

        return on_startup

    @property
    def on_shutdown(self) -> list[t.Callable[[], None]]:
        on_shutdown = [self.bento_service.on_grpc_server_shutdown]
        for runner in self.bento_service.runners:
            on_shutdown.append(runner.destroy)

        return on_shutdown

    def __call__(self) -> GRPCServer:
        return GRPCServer(
            server=self.server,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
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
        from .grpc.interceptors import ExceptionHandlerInterceptor

        # # TODO: add access log, tracing, prometheus interceptors.
        interceptors: list[aio.ServerInterceptor] = [ExceptionHandlerInterceptor()]

        # add users-defined interceptors.
        interceptors.extend(
            [interceptor() for interceptor in self.bento_service.interceptors]
        )
        return interceptors
