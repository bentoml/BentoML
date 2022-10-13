from __future__ import annotations

import os
import sys
import typing as t
import asyncio
import logging
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from simple_di import inject
from simple_di import Provide

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import import_generated_stubs

from ...utils import LazyLoader
from ...utils import cached_property
from ...utils import resolve_user_filepath
from ...configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health_pb2 as pb_health
    from grpc_health.v1 import health_pb2_grpc as services_health

    from bentoml.grpc.v1alpha1 import service_pb2_grpc as services

    from .servicer import Servicer
else:
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


def _load_from_file(p: str) -> bytes:
    rp = resolve_user_filepath(p, ctx=None)
    with open(rp, "rb") as f:
        return f.read()


# NOTE: we are using the internal aio._server.Server (which is initialized with aio.server)
class Server(aio._server.Server):
    """An async implementation of a gRPC server."""

    @inject
    def __init__(
        self,
        servicer: Servicer,
        bind_address: str,
        max_message_length: int
        | None = Provide[BentoMLContainer.grpc.max_message_length],
        maximum_concurrent_rpcs: int
        | None = Provide[BentoMLContainer.grpc.maximum_concurrent_rpcs],
        enable_reflection: bool = False,
        enable_channelz: bool = False,
        max_concurrent_streams: int | None = None,
        migration_thread_pool_workers: int = 1,
        ssl_certfile: str | None = None,
        ssl_keyfile: str | None = None,
        ssl_ca_certs: str | None = None,
        graceful_shutdown_timeout: float | None = None,
        compression: grpc.Compression | None = None,
    ):
        self.servicer = servicer
        self.max_message_length = max_message_length
        self.max_concurrent_streams = max_concurrent_streams
        self.bind_address = bind_address
        self.enable_reflection = enable_reflection
        self.enable_channelz = enable_channelz
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_ca_certs = ssl_ca_certs

        if not bool(self.servicer):
            self.servicer.load()
        assert self.servicer.loaded

        super().__init__(
            # Note that the max_workers are used inside ThreadPoolExecutor.
            # This ThreadPoolExecutor are used by aio.Server() to execute non-AsyncIO RPC handlers.
            # Setting it to 1 makes it thread-safe for sync APIs.
            thread_pool=ThreadPoolExecutor(max_workers=migration_thread_pool_workers),
            generic_handlers=() if self.handlers is None else self.handlers,
            interceptors=self.servicer.interceptors_stack,
            options=self.options,
            # maximum_concurrent_rpcs defines the maximum number of concurrent RPCs this server
            # will service before returning RESOURCE_EXHAUSTED status.
            # Set to None will indicate no limit.
            maximum_concurrent_rpcs=maximum_concurrent_rpcs,
            compression=compression,
        )

    @property
    def options(self) -> grpc.aio.ChannelArgumentType:
        options: grpc.aio.ChannelArgumentType = []

        if sys.platform != "win32":
            # https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/grpc_types.h#L294
            # Eventhough GRPC_ARG_ALLOW_REUSEPORT is set to 1 by default, we want still
            # want to explicitly set it to 1 so that we can spawn multiple gRPC servers in
            # production settings.
            options.append(("grpc.so_reuseport", 1))
        if self.max_concurrent_streams:
            options.append(("grpc.max_concurrent_streams", self.max_concurrent_streams))
        if self.enable_channelz:
            options.append(("grpc.enable_channelz", 1))
        if self.max_message_length:
            options.extend(
                (
                    # grpc.max_message_length this is a deprecated options, for backward compatibility
                    ("grpc.max_message_length", self.max_message_length),
                    ("grpc.max_receive_message_length", self.max_message_length),
                    ("grpc.max_send_message_length", self.max_message_length),
                )
            )

        return tuple(options)

    @property
    def handlers(self) -> t.Sequence[grpc.GenericRpcHandler] | None:
        # Note that currently BentoML doesn't provide any specific
        # handlers for gRPC. If users have any specific handlers,
        # BentoML will pass it through to grpc.aio.Server
        return self.servicer.bento_service.grpc_handlers

    @cached_property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def run(self) -> None:
        try:
            self.loop.run_until_complete(self.serve())
        finally:
            try:
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.shutdown())
                )
            except Exception as e:  # pylint: disable=broad-except
                raise RuntimeError(f"Server failed unexpectedly: {e}") from None

    def configure_port(self, addr: str):
        if self.ssl_certfile:
            client_auth = False
            ca_cert = None
            assert (
                self.ssl_keyfile
            ), "'ssl_keyfile' is required when 'ssl_certfile' is provided."
            if self.ssl_ca_certs is not None:
                client_auth = True
                ca_cert = _load_from_file(self.ssl_ca_certs)
            server_credentials = grpc.ssl_server_credentials(
                (
                    (
                        _load_from_file(self.ssl_keyfile),
                        _load_from_file(self.ssl_certfile),
                    ),
                ),
                root_certificates=ca_cert,
                require_client_auth=client_auth,
            )

            self.add_secure_port(addr, server_credentials)
        else:
            self.add_insecure_port(addr)

    async def serve(self) -> None:
        self.configure_port(self.bind_address)
        await self.startup()
        await self.wait_for_termination()

    async def startup(self) -> None:
        from bentoml.exceptions import MissingDependencyException

        # Running on_startup callback.
        await self.servicer.startup()
        # register bento servicer
        services.add_BentoServiceServicer_to_server(self.servicer.bento_servicer, self)
        services_health.add_HealthServicer_to_server(
            self.servicer.health_servicer, self
        )

        service_names = self.servicer.service_names
        # register custom servicer
        for (
            user_servicer,
            add_servicer_fn,
            user_service_names,
        ) in self.servicer.mount_servicers:
            add_servicer_fn(user_servicer(), self)
            service_names += tuple(user_service_names)
        if self.enable_channelz:
            try:
                from grpc_channelz.v1 import channelz
            except ImportError:
                raise MissingDependencyException(
                    "'--debug' is passed, which requires 'grpcio-channelz' to be installed. Install with 'pip install bentoml[grpc-channelz]'."
                ) from None
            if "GRPC_TRACE" not in os.environ:
                logger.debug(
                    "channelz is enabled, while GRPC_TRACE is not set. No channel tracing will be recorded."
                )
            channelz.add_channelz_servicer(self)
        if self.enable_reflection:
            try:
                # reflection is required for health checking to work.
                from grpc_reflection.v1alpha import reflection
            except ImportError:
                raise MissingDependencyException(
                    "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with 'pip install bentoml[grpc-reflection]'."
                ) from None
            service_names += (reflection.SERVICE_NAME,)
            reflection.enable_server_reflection(service_names, self)
        # mark all services as healthy
        for service in service_names:
            await self.servicer.health_servicer.set(
                service, pb_health.HealthCheckResponse.SERVING  # type: ignore (no types available)
            )
        await self.start()

    async def shutdown(self):
        # Running on_startup callback.
        await self.servicer.shutdown()
        await self.stop(grace=self.graceful_shutdown_timeout)
        await self.servicer.health_servicer.enter_graceful_shutdown()
        self.loop.stop()
