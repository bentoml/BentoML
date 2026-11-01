from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import typing as t
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING

from simple_di import Provide
from simple_di import inject

from _bentoml_impl.server.grpc.servicer.v1 import create_bento_servicer
from _bentoml_sdk import Service
from _bentoml_sdk.service import set_current_service
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.context import ServiceContext as Context
from bentoml._internal.utils.lazy_loader import LazyLoader
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml.grpc.utils import LATEST_PROTOCOL_VERSION
from bentoml.grpc.utils import import_generated_stubs
from bentoml.grpc.utils import import_grpc
from bentoml.grpc.utils import load_from_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health
    from grpc_health.v1 import health_pb2 as pb_health
    from grpc_health.v1 import health_pb2_grpc as services_health

    from bentoml.grpc.types import Interceptors
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
        exc_msg=health_exception_msg,
    )


class Server(aio._server.Server):
    """Async gRPC server for ``@bentoml.service()`` services."""

    @inject
    def __init__(
        self,
        bento_service: Service[t.Any],
        bind_address: str,
        max_message_length: int | None = Provide[
            BentoMLContainer.grpc.max_message_length
        ],
        maximum_concurrent_rpcs: int | None = Provide[
            BentoMLContainer.grpc.maximum_concurrent_rpcs
        ],
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
        if protocol_version != "v1":
            raise BentoMLException(
                f"@bentoml.service() gRPC serving only supports protocol v1, got {protocol_version!r}"
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        pb, _ = import_generated_stubs("v1")

        self.bento_service = bento_service
        self.servicer = create_bento_servicer(bento_service)
        self._service_instance: t.Any | None = None

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

        self.health_servicer = health.aio.HealthServicer()
        self.service_names = tuple(
            service.full_name for service in pb.DESCRIPTOR.services_by_name.values()
        ) + (health.SERVICE_NAME,)

        super().__init__(
            thread_pool=ThreadPoolExecutor(max_workers=migration_thread_pool_workers),
            generic_handlers=() if self.handlers is None else self.handlers,
            interceptors=list(map(lambda x: x(), self.interceptors)),
            options=self.options,
            maximum_concurrent_rpcs=maximum_concurrent_rpcs,
            compression=compression,
        )

    @property
    def options(self) -> grpc.aio.ChannelArgumentType:
        options: grpc.aio.ChannelArgumentType = []
        if sys.platform != "win32":
            options.append(("grpc.so_reuseport", 1))
        if self.max_concurrent_streams:
            options.append(("grpc.max_concurrent_streams", self.max_concurrent_streams))
        if self.enable_channelz:
            options.append(("grpc.enable_channelz", 1))
        if self.max_message_length:
            options.extend(
                (
                    ("grpc.max_message_length", self.max_message_length),
                    ("grpc.max_receive_message_length", self.max_message_length),
                    ("grpc.max_send_message_length", self.max_message_length),
                )
            )
        return tuple(options)

    @property
    def interceptors(self) -> Interceptors:
        from bentoml.grpc.interceptors.opentelemetry import (
            AsyncOpenTelemetryServerInterceptor,
        )

        interceptors: Interceptors = [AsyncOpenTelemetryServerInterceptor]
        if BentoMLContainer.api_server_config.metrics.enabled.get():
            from bentoml.grpc.interceptors.prometheus import PrometheusServerInterceptor

            interceptors.append(PrometheusServerInterceptor)
        if BentoMLContainer.api_server_config.logging.access.enabled.get():
            from bentoml.grpc.interceptors.access import AccessLogServerInterceptor

            access_logger = logging.getLogger("bentoml.access")
            if access_logger.getEffectiveLevel() <= logging.INFO:
                interceptors.append(AccessLogServerInterceptor)
        return interceptors

    @property
    def handlers(self) -> t.Sequence[grpc.GenericRpcHandler] | None:
        return None

    @cached_property
    def loop(self) -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
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

    @cached_property
    def context(self) -> Context:
        return self.bento_service.context

    def configure_port(self, addr: str) -> None:
        if self.ssl_certfile:
            client_auth = False
            ca_cert = None
            assert self.ssl_keyfile, (
                "'ssl_keyfile' is required when 'ssl_certfile' is provided."
            )
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

    async def _create_instance(self) -> None:
        self._service_instance = self.bento_service()
        logger.info("Service %s initialized", self.bento_service.name)
        for name in dir(self.bento_service.inner):
            member = getattr(self.bento_service.inner, name)
            if (
                not name.startswith("__")
                and callable(member)
                and getattr(member, "__bentoml_startup_hook__", False)
            ):
                logger.info("Running startup hook: %s", name)
                result = getattr(self._service_instance, name)()
                if inspect.isawaitable(result):
                    await result
        set_current_service(self._service_instance)
        self.servicer.set_instance(self._service_instance)
        await asyncio.gather(
            *(
                real.__aenter__()
                for dep_name in self.bento_service.dependencies
                if hasattr(
                    (real := getattr(self._service_instance, dep_name)), "__aenter__"
                )
            )
        )

    async def startup(self) -> None:
        _, services = import_generated_stubs("v1")

        await self._create_instance()

        services.add_BentoServiceServicer_to_server(self.servicer, self)
        services_health.add_HealthServicer_to_server(self.health_servicer, self)

        service_names = self.service_names
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
                from grpc_reflection.v1alpha import reflection
            except ImportError:
                raise MissingDependencyException(
                    "reflection is enabled, which requires 'grpcio-reflection' to be installed. Install with 'pip install bentoml[grpc-reflection]'."
                ) from None
            service_names += (reflection.SERVICE_NAME,)
            reflection.enable_server_reflection(service_names, self)
        for service in service_names:
            await self.health_servicer.set(
                service,
                pb_health.HealthCheckResponse.SERVING,  # type: ignore (no types available)
            )
        await self.start()

    async def shutdown(self) -> None:
        from _bentoml_sdk.service.dependency import cleanup

        if self._service_instance is not None:
            for name in dir(self.bento_service.inner):
                member = getattr(self.bento_service.inner, name)
                if (
                    not name.startswith("__")
                    and callable(member)
                    and getattr(member, "__bentoml_shutdown_hook__", False)
                ):
                    result = getattr(self._service_instance, name)()
                    if inspect.isawaitable(result):
                        await result
            await cleanup()
            self._service_instance = None
            set_current_service(None)

        await self.stop(grace=self.graceful_shutdown_timeout)
        await self.health_servicer.enter_graceful_shutdown()
        self.loop.stop()
