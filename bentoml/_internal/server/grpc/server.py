from __future__ import annotations

import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING

import grpc
from grpc import aio

from ...utils import LazyLoader
from ...utils import cached_property

logger = logging.getLogger(__name__)

if TYPE_CHECKING:

    from grpc_health.v1 import health_pb2 as pb_health
    from grpc_health.v1 import health_pb2_grpc as health_services

    from bentoml.grpc.v1 import service_pb2_grpc as services

    from .config import Config
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, services = import_generated_stubs()
    health_exception_msg = "'grpcio-health-checking' is required for using health checking endpoints. Install with 'pip install grpcio-health-checking'."
    pb_health = LazyLoader(
        "pb_health",
        globals(),
        "grpc_health.v1.health_pb2",
        exc_msg=health_exception_msg,
    )
    health_services = LazyLoader(
        "health_services",
        globals(),
        "grpc_health.v1.health_pb2_grpc",
        exc_msg=health_exception_msg,
    )


class Server:
    """An Uvicorn-like implementation for async gRPC server."""

    def __init__(self, config: Config):
        self.config = config
        self.servicer = config.servicer

        # define a cleanup future list
        self.cleanup_tasks: list[t.Coroutine[t.Any, t.Any, None]] = []

    @cached_property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def run(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        if not bool(self.servicer):
            self.servicer.load()

        self.server = aio.server(
            migration_thread_pool=ThreadPoolExecutor(
                max_workers=self.config.migration_thread_pool_workers
            ),
            options=self.config.options,
            maximum_concurrent_rpcs=self.config.maximum_concurrent_rpcs,
            handlers=self.config.handlers,
            interceptors=self.servicer.interceptors_stack,
        )

        try:
            self.loop.run_until_complete(self.serve())
        finally:
            try:
                if self.cleanup_tasks:
                    self.loop.run_until_complete(*self.cleanup_tasks)
                    self.loop.close()
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"Server failed unexpectedly. enable GRPC_VERBOSITY=debug for more information: {e}"
                ) from e

    async def serve(self) -> None:
        self.add_insecure_port(self.config.bind_address)

        await self.startup()

        self.cleanup_tasks.append(self.shutdown())

        await self.wait_for_termination()

    async def startup(self) -> None:
        from bentoml.exceptions import MissingDependencyException

        # Running on_startup callback.
        await self.servicer.startup()

        # register bento servicer
        services.add_BentoServiceServicer_to_server(
            self.servicer.bento_servicer, self.server
        )
        health_services.add_HealthServicer_to_server(
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
                    "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with 'pip install grpcio-relfection'."
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
