from __future__ import annotations

import os
import sys
import typing as t
import asyncio
import inspect
import logging
from typing import TYPE_CHECKING
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from simple_di import inject
from simple_di import Provide

from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import import_generated_stubs

from ..utils import LazyLoader
from ..utils import cached_property
from ..utils import resolve_user_filepath
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health
    from grpc_health.v1 import health_pb2 as pb_health
    from grpc_health.v1 import health_pb2_grpc as services_health

    from ..service import Service
    from ...grpc.types import Interceptors

    OnStartup = list[t.Callable[[], t.Union[None, t.Coroutine[t.Any, t.Any, None]]]]

else:
    grpc, aio = import_grpc()
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
    health = LazyLoader(
        "health",
        globals(),
        "grpc_health.v1.health",
        exc_msg="'grpcio-health-checking' is required for using health checking endpoints. Install with 'pip install grpcio-health-checking'.",
    )


def load_from_file(p: str) -> bytes:
    rp = resolve_user_filepath(p, ctx=None)
    with open(rp, "rb") as f:
        return f.read()


# NOTE: we are using the internal aio._server.Server (which is initialized with aio.server)
class Server(aio._server.Server):
    """An async implementation of a gRPC server."""

    @inject
    def __init__(
        self,
        bento_service: Service,
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
        protocol_version: str = LATEST_PROTOCOL_VERSION,
    ):
        pb, _ = import_generated_stubs(protocol_version)

        self.bento_service = bento_service
        self.servicer = bento_service.get_grpc_servicer(protocol_version)

        # options
        self.max_message_length = max_message_length
        self.max_concurrent_streams = max_concurrent_streams
        self.bind_address = bind_address
        self.enable_reflection = enable_reflection
        self.enable_channelz = enable_channelz
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_ca_certs = ssl_ca_certs
        self.protocol_version = protocol_version

        # Create a health check servicer. We use the non-blocking implementation
        # to avoid thread starvation.
        self.health_servicer = health.aio.HealthServicer()

        self.mount_servicers = self.bento_service.mount_servicers

        self.service_names = tuple(
            service.full_name for service in pb.DESCRIPTOR.services_by_name.values()
        ) + (health.SERVICE_NAME,)

        super().__init__(
            # Note that the max_workers are used inside ThreadPoolExecutor.
            # This ThreadPoolExecutor are used by aio.Server() to execute non-AsyncIO RPC handlers.
            # Setting it to 1 makes it thread-safe for sync APIs.
            thread_pool=ThreadPoolExecutor(max_workers=migration_thread_pool_workers),
            generic_handlers=() if self.handlers is None else self.handlers,
            interceptors=list(map(lambda x: x(), self.interceptors)),
            options=self.options,
            # maximum_concurrent_rpcs defines the maximum number of concurrent RPCs this server
            # will service before returning RESOURCE_EXHAUSTED status.
            # Set to None will indicate no limit.
            maximum_concurrent_rpcs=maximum_concurrent_rpcs,
            compression=compression,
        )

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
    def interceptors(self) -> Interceptors:
        # Note that order of interceptors is important here.

        from ...grpc.interceptors.opentelemetry import (
            AsyncOpenTelemetryServerInterceptor,
        )

        interceptors: Interceptors = [AsyncOpenTelemetryServerInterceptor]

        if BentoMLContainer.api_server_config.metrics.enabled.get():
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

    @property
    def handlers(self) -> t.Sequence[grpc.GenericRpcHandler] | None:
        # Note that currently BentoML doesn't provide any specific
        # handlers for gRPC. If users have any specific handlers,
        # BentoML will pass it through to grpc.aio.Server
        return self.bento_service.grpc_handlers

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
                ca_cert = load_from_file(self.ssl_ca_certs)
            server_credentials = grpc.ssl_server_credentials(
                (
                    (
                        load_from_file(self.ssl_keyfile),
                        load_from_file(self.ssl_certfile),
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

    async def startup(self) -> None:
        from ...exceptions import MissingDependencyException

        _, services = import_generated_stubs(self.protocol_version)

        # Running on_startup callback.
        for handler in self.on_startup:
            out = handler()
            if inspect.isawaitable(out):
                await out

        # register bento servicer
        services.add_BentoServiceServicer_to_server(self.servicer, self)
        services_health.add_HealthServicer_to_server(self.health_servicer, self)

        service_names = self.service_names
        # register custom servicer
        for (
            user_servicer,
            add_servicer_fn,
            user_service_names,
        ) in self.mount_servicers:
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
            await self.health_servicer.set(
                service, pb_health.HealthCheckResponse.SERVING  # type: ignore (no types available)
            )
        await self.start()

    @property
    def on_shutdown(self) -> list[t.Callable[[], None]]:
        on_shutdown = [self.bento_service.on_grpc_server_shutdown]
        for runner in self.bento_service.runners:
            on_shutdown.append(runner.destroy)

        return on_shutdown

    async def shutdown(self):
        # Running on_startup callback.
        for handler in self.on_shutdown:
            out = handler()
            if inspect.isawaitable(out):
                await out

        await self.stop(grace=self.graceful_shutdown_timeout)
        await self.health_servicer.enter_graceful_shutdown()
        self.loop.stop()
