from __future__ import annotations

import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING

import grpc
from grpc import aio

from bentoml.exceptions import MissingDependencyException

from ...utils import LazyLoader
from ...utils import cached_property

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grpc_health.v1 import health
    from grpc_health.v1 import health_pb2
    from grpc_health.v1 import health_pb2_grpc
    from google.protobuf.descriptor import ServiceDescriptor

    from bentoml.grpc.v1 import service_pb2 as pb
    from bentoml.grpc.v1 import service_pb2_grpc as services
    from bentoml.grpc.types import AddServicerFn
    from bentoml.grpc.types import ServicerClass
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, services = import_generated_stubs()
    health = LazyLoader("health", globals(), "grpc_health.v1.health")
    health_pb2 = LazyLoader("health_pb2", globals(), "grpc_health.v1.health_pb2")
    health_pb2_grpc = LazyLoader(
        "health_pb2_grpc", globals(), "grpc_health.v1.health_pb2_grpc"
    )


class GRPCServer:
    """An ASGI-like implementation for async gRPC server."""

    def __init__(
        self,
        server: aio.Server,
        on_startup: t.Sequence[t.Callable[[], t.Any]] | None = None,
        on_shutdown: t.Sequence[t.Callable[[], t.Any]] | None = None,
        mount_servicers: t.Sequence[
            tuple[ServicerClass, AddServicerFn, list[ServiceDescriptor]]
        ]
        | None = None,
        *,
        _grace_period: int = 5,
        _bento_servicer: services.BentoServiceServicer,
        _health_servicer: health.aio.HealthServicer,
    ):
        self._bento_servicer = _bento_servicer
        self._health_servicer = _health_servicer
        self._grace_period = _grace_period

        self.server = server

        # define a cleanup future list
        self._cleanup: list[t.Coroutine[t.Any, t.Any, None]] = []

        self.on_startup = [] if on_startup is None else list(on_startup)
        self.on_shutdown = [] if on_shutdown is None else list(on_shutdown)
        self.mount_servicers = [] if mount_servicers is None else list(mount_servicers)

    @cached_property
    def _loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def run(self, bind_addr: str) -> None:
        try:
            self._loop.run_until_complete(self.serve(bind_addr=bind_addr))
        finally:
            try:
                if self._cleanup:
                    self._loop.run_until_complete(*self._cleanup)
                    self._loop.close()
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(
                    f"Server failed unexpectedly. enable GRPC_VERBOSITY=debug for more information: {e}"
                ) from e

    async def serve(self, bind_addr: str) -> None:
        self.add_insecure_port(bind_addr)

        await self.startup()

        self._cleanup.append(self.shutdown())

        await self.wait_for_termination()

    async def startup(self) -> None:
        try:
            # reflection is required for health checking to work.
            from grpc_reflection.v1alpha import reflection
        except ImportError:
            raise MissingDependencyException(
                "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with `pip install 'grpcio-relfection'.`"
            )

        # Running on_startup callback.
        for handler in self.on_startup:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

        # register bento servicer
        services.add_BentoServiceServicer_to_server(
            self._bento_servicer, self.server  # type: ignore (unfinished async types)
        )
        health_pb2_grpc.add_HealthServicer_to_server(self._health_servicer, self.server)

        service_name = tuple(
            service.full_name for service in pb.DESCRIPTOR.services_by_name.values()
        )

        # register custom servicer
        for servicer, add_servicer_fn, service_descriptor in self.mount_servicers:
            # TODO: Annotated types are not contravariant
            add_servicer_fn(servicer(), self.server)
            service_name += tuple(service.full_name for service in service_descriptor)

        service_name += (health.SERVICE_NAME, reflection.SERVICE_NAME)
        reflection.enable_server_reflection(service_name, self.server)

        # mark all services as healthy
        for service in service_name:
            await self._health_servicer.set(
                service, health_pb2.HealthCheckResponse.SERVING  # type: ignore (no types available)
            )

        await self.server.start()

    async def shutdown(self):
        # Running on_startup callback.
        for handler in self.on_shutdown:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

        await self.server.stop(grace=self._grace_period)
        await self._health_servicer.enter_graceful_shutdown()

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
