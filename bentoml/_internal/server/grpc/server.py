from __future__ import annotations

import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING

from ...utils import LazyLoader
from ...utils import cached_property

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health_pb2 as pb_health
    from grpc_health.v1 import health_pb2_grpc as services_health
    from typing_extensions import Self

    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services

    from .config import Config
else:
    from bentoml.grpc.utils import import_grpc
    from bentoml.grpc.utils import import_generated_stubs
    from bentoml._internal.utils import LazyLoader

    grpc, aio = import_grpc()
    _, services = import_generated_stubs()
    health_exception_msg = "'grpcio-health-checking' is required for using health checking endpoints. Install with 'pip install grpcio-health-checking'."
    pb_health = LazyLoader(
        "pb_health",
        globals(),
        "grpc_health.v1.health_pb2",
        exc_msg=health_exception_msg,
    )
    services_health = LazyLoader(
        "services_health",
        globals(),
        "grpc_health.v1.health_pb2_grpc",
        exc_msg=health_exception_msg,
    )


class Server:
    """An async implementation of a gRPC server."""

    def __init__(self, config: Config):
        self.config = config
        self.servicer = config.servicer
        self.loaded = False

    @cached_property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def load(self) -> Self:
        from concurrent.futures import ThreadPoolExecutor

        assert not self.loaded
        if not bool(self.servicer):
            self.servicer.load()
        assert self.servicer.loaded

        self.server = aio.server(
            migration_thread_pool=ThreadPoolExecutor(
                max_workers=self.config.migration_thread_pool_workers
            ),
            options=self.config.options,
            maximum_concurrent_rpcs=self.config.maximum_concurrent_rpcs,
            handlers=self.config.handlers,
            interceptors=self.servicer.interceptors_stack,
        )
        self.loaded = True

        return self

    def run(self) -> None:
        if not self.loaded:
            self.load()
        assert self.loaded

        try:
            self.loop.run_until_complete(self.serve())
        finally:
            try:
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.shutdown())
                )
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(f"Server failed unexpectedly: {e}") from None

    async def serve(self) -> None:
        self.add_insecure_port(self.config.bind_address)
        await self.startup()
        await self.wait_for_termination()

    async def startup(self) -> None:
        from bentoml.exceptions import MissingDependencyException

        # Running on_startup callback.
        await self.servicer.startup()
        # register bento servicer
        services.add_BentoServiceServicer_to_server(
            self.servicer.bento_servicer, self.server
        )
        services_health.add_HealthServicer_to_server(
            self.servicer.health_servicer, self.server
        )

        service_names = self.servicer.service_names
        # register custom servicer
        for (
            user_servicer,
            add_servicer_fn,
            user_service_names,
        ) in self.servicer.mount_servicers:
            add_servicer_fn(user_servicer(), self.server)
            service_names += tuple(user_service_names)

        if self.config.enable_reflection:
            try:
                # reflection is required for health checking to work.
                from grpc_reflection.v1alpha import reflection
            except ImportError:
                raise MissingDependencyException(
                    "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with 'pip install grpcio-reflection'."
                )
            service_names += (reflection.SERVICE_NAME,)
            reflection.enable_server_reflection(service_names, self.server)

        # mark all services as healthy
        for service in service_names:
            await self.servicer.health_servicer.set(
                service, pb_health.HealthCheckResponse.SERVING  # type: ignore (no types available)
            )
        await self.server.start()

    async def shutdown(self):
        # Running on_startup callback.
        await self.servicer.shutdown()
        await self.server.stop(grace=self.config.graceful_shutdown_timeout)
        await self.servicer.health_servicer.enter_graceful_shutdown()
        self.loop.stop()

    async def wait_for_termination(self, timeout: int | None = None) -> bool:
        return await self.server.wait_for_termination(timeout=timeout)

    def add_insecure_port(self, address: str) -> int:
        return self.server.add_insecure_port(address)

    def add_secure_port(self, address: str, credentials: grpc.ServerCredentials) -> int:
        return self.server.add_secure_port(address, credentials)

    def add_generic_rpc_handlers(
        self, generic_rpc_handlers: t.Sequence[grpc.GenericRpcHandler]
    ) -> None:
        self.server.add_generic_rpc_handlers(generic_rpc_handlers)
