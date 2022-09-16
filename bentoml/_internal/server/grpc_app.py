from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from functools import partial

from simple_di import inject
from simple_di import Provide

from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:

    from bentoml.grpc.types import Interceptors

    from ..service import Service
    from .grpc.servicer import Servicer

    OnStartup = list[t.Callable[[], None | t.Coroutine[t.Any, t.Any, None]]]


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
        enable_metrics: bool = Provide[
            BentoMLContainer.api_server_config.metrics.enabled
        ],
    ) -> None:
        self.bento_service = bento_service
        self.enable_metrics = enable_metrics

    @property
    def on_startup(self) -> OnStartup:
        on_startup: OnStartup = [self.bento_service.on_grpc_server_startup]
        if BentoMLContainer.development_mode.get():
            for runner in self.bento_service.runners:
                on_startup.append(partial(runner.init_local, quiet=True))
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

    def __call__(self) -> Servicer:
        from .grpc import Servicer

        return Servicer(
            self.bento_service,
            on_startup=self.on_startup,
            on_shutdown=self.on_shutdown,
            mount_servicers=self.bento_service.mount_servicers,
            interceptors=self.interceptors,
        )

    @property
    def interceptors(self) -> Interceptors:
        # Note that order of interceptors is important here.

        from bentoml.grpc.interceptors.opentelemetry import (
            AsyncOpenTelemetryServerInterceptor,
        )

        interceptors: Interceptors = [AsyncOpenTelemetryServerInterceptor]

        if self.enable_metrics:
            from bentoml.grpc.interceptors.prometheus import PrometheusServerInterceptor

            interceptors.append(PrometheusServerInterceptor)

        if BentoMLContainer.api_server_config.logging.access.enabled.get():
            from bentoml.grpc.interceptors.access import AccessLogServerInterceptor

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                interceptors.append(AccessLogServerInterceptor)

        # add users-defined interceptors.
        interceptors.extend(self.bento_service.interceptors)

        return interceptors
