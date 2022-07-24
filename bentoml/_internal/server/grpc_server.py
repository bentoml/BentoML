from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from concurrent import futures

import attr
import grpc
from simple_di import Provide

from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..service import Service

logger = logging.getLogger(__name__)


@attr.define(init=False)
class GRPCServer:
    """
    A light wrapper around grpc.aio.Server.
    """

    # Set a thread pool size of 10 as default. Subject to change.
    _thread_pool_size = 10
    # Set a grace period for the server to stop, default to 5.
    _grace_period = 5

    interceptors: list[grpc.aio.ServerInterceptor]

    def __init__(
        self,
        bento_service: Service,
        interceptors: t.List[t.Type[grpc.aio.ServerInterceptor]],
    ):
        self.bento_service = bento_service
        self.__attrs_init__(  # type: ignore (unfinished attrs init type)
            interceptors=[interceptor() for interceptor in interceptors]
        )

    def startup(self) -> None:
        if BentoMLContainer.development_mode.get():
            for runner in self.bento_service.runners:
                runner.init_local(quiet=True)
        else:
            for runner in self.bento_service.runners:
                runner.init_client()

    def shutdown(self) -> None:
        for runner in self.bento_service.runners:
            runner.destroy()

    @property
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

    def __call__(self) -> GRPCServer:
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(self._thread_pool_size),
            interceptors=self.interceptors,
            options=self.options,
        )

        return self

    def add_insecure_port(self, address: str) -> int:
        return self.server.add_insecure_port(address)

    def add_secure_port(
        self, address: str, server_credentials: grpc.ServerCredentials
    ) -> int:
        return self.server.add_secure_port(address, server_credentials)

    def add_generic_rpc_handlers(
        self, generic_rpc_handlers: t.Sequence[grpc.GenericRpcHandler]
    ) -> None:
        self.server.add_generic_rpc_handlers(generic_rpc_handlers)

    async def register_servicer(self):
        from bentoml._internal.configuration import get_debug_mode
        from bentoml.grpc.v1.service_pb2_grpc import add_BentoServiceServicer_to_server

        from .grpc.servicer import add_health_servicer
        from .grpc.servicer import add_bentoservice_servicer

        # add health checking servicer.
        await add_health_servicer(self.server, enable_reflection=get_debug_mode())

        add_BentoServiceServicer_to_server(
            add_bentoservice_servicer(self)(), self.server
        )

    async def start(self):
        self.startup()

        # register all servicer.
        await self.register_servicer()

        await self.server.start()
        logger.debug("GRPC server started.")

    async def stop(self):
        self.shutdown()

        await self.server.stop(grace=self._grace_period)
        logger.debug("GRPC server stopped.")

    async def wait_for_termination(self, timeout: int | None = None) -> bool:
        return await self.server.wait_for_termination(timeout=timeout)
