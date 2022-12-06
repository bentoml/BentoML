from __future__ import annotations

import typing as t
import asyncio
import logging
import importlib
from typing import TYPE_CHECKING
from functools import partial

from simple_di import inject
from simple_di import Provide

from ..utils import LazyLoader
from ...grpc.utils import import_generated_stubs
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types import ModuleType

    from grpc_health.v1 import health

    from bentoml.grpc.types import Interceptors

    from ..service import Service
    from ...grpc.v1 import service_pb2_grpc as services

    class ServicerModule(ModuleType):
        @staticmethod
        def create_bento_servicer(service: Service) -> services.BentoServiceServicer:
            ...

    OnStartup = list[t.Callable[[], t.Union[None, t.Coroutine[t.Any, t.Any, None]]]]
else:
    health = LazyLoader(
        "health",
        globals(),
        "grpc_health.v1.health",
        exc_msg="'grpcio-health-checking' is required for using health checking endpoints. Install with 'pip install grpcio-health-checking'.",
    )


class GrpcServicerFactory:
    """
    GrpcServicerFactory creates an async gRPC API server based on APIs defined with a BentoService via BentoService.apis.
    This is a light wrapper around GrpcServer with addition to `on_startup` and `on_shutdown` hooks.

    Note that even though the code are similar with BaseAppFactory, gRPC protocol is different from ASGI.
    """

    _cached_module = None

    @inject
    def __init__(
        self,
        bento_service: Service,
        *,
        enable_metrics: bool = Provide[
            BentoMLContainer.api_server_config.metrics.enabled
        ],
        protocol_version: str = LATEST_PROTOCOL_VERSION,
    ) -> None:
        pb, _ = import_generated_stubs(protocol_version)

        self.bento_service = bento_service
        self.enable_metrics = enable_metrics
        self.protocol_version = protocol_version
        self.interceptors_stack = list(map(lambda x: x(), self.interceptors))

        self.bento_servicer = self._servicer_module.create_bento_servicer(
            self.bento_service
        )
        self.mount_servicers = self.bento_service.mount_servicers

        # Create a health check servicer. We use the non-blocking implementation
        # to avoid thread starvation.
        self.health_servicer = health.aio.HealthServicer()

        self.service_names = tuple(
            service.full_name for service in pb.DESCRIPTOR.services_by_name.values()
        ) + (health.SERVICE_NAME,)

    @inject
    async def wait_for_runner_ready(
        self,
        *,
        check_interval: int = Provide[
            BentoMLContainer.api_server_config.runner_probe.period
        ],
    ):
        if BentoMLContainer.api_server_config.runner_probe.enabled.get():
            logger.info("Waiting for runners to be ready...")
            logger.debug("Current runners: %r", self.bento_service.runners)

            while True:
                try:
                    runner_statuses = (
                        runner.runner_handle_is_ready()
                        for runner in self.bento_service.runners
                    )
                    runners_ready = all(await asyncio.gather(*runner_statuses))

                    if runners_ready:
                        break
                except ConnectionError as e:
                    logger.debug("[%s] Retrying ...", e)

                await asyncio.sleep(check_interval)

            logger.info("All runners ready.")

    @property
    def on_startup(self) -> OnStartup:
        on_startup: OnStartup = [self.bento_service.on_grpc_server_startup]
        if BentoMLContainer.development_mode.get():
            for runner in self.bento_service.runners:
                on_startup.append(partial(runner.init_local, quiet=True))
        else:
            for runner in self.bento_service.runners:
                on_startup.append(runner.init_client)

        on_startup.append(self.wait_for_runner_ready)
        return on_startup

    @property
    def on_shutdown(self) -> list[t.Callable[[], None]]:
        on_shutdown = [self.bento_service.on_grpc_server_shutdown]
        for runner in self.bento_service.runners:
            on_shutdown.append(runner.destroy)

        return on_shutdown

    @property
    def _servicer_module(self) -> ServicerModule:
        if self._cached_module is None:
            object.__setattr__(
                self,
                "_cached_module",
                importlib.import_module(
                    f".grpc.servicer.{self.protocol_version}",
                    package="bentoml._internal.server",
                ),
            )
        assert self._cached_module is not None
        return self._cached_module

    @property
    def interceptors(self) -> Interceptors:
        # Note that order of interceptors is important here.

        from ...grpc.interceptors.opentelemetry import (
            AsyncOpenTelemetryServerInterceptor,
        )

        interceptors: Interceptors = [AsyncOpenTelemetryServerInterceptor]

        if self.enable_metrics:
            from ...grpc.interceptors.prometheus import PrometheusServerInterceptor

            interceptors.append(PrometheusServerInterceptor)

        if BentoMLContainer.api_server_config.logging.access.enabled.get():
            from ...grpc.interceptors.access import AccessLogServerInterceptor

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                interceptors.append(AccessLogServerInterceptor)

        # add users-defined interceptors.
        interceptors.extend(self.bento_service.interceptors)

        return interceptors
