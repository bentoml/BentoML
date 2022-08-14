from __future__ import annotations

import typing as t
import asyncio
import logging

import grpc
from grpc import aio

from ...utils import cached_property

logger = logging.getLogger(__name__)


class GRPCServer:
    """An ASGI-like implementation for async gRPC server."""

    def __init__(
        self,
        server: aio.Server,
        on_startup: t.Sequence[t.Callable[[], t.Any]] | None = None,
        on_shutdown: t.Sequence[t.Callable[[], t.Any]] | None = None,
        *,
        _grace_period: int = 5,
    ):
        self._grace_period = _grace_period

        self.server = server

        # define a cleanup future list
        self._cleanup: list[t.Coroutine[t.Any, t.Any, None]] = []

        self.on_startup = [] if on_startup is None else list(on_startup)
        self.on_shutdown = [] if on_shutdown is None else list(on_shutdown)

    @cached_property
    def _loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def run(self, bind_addr: str) -> None:
        try:
            self._loop.run_until_complete(self.serve(bind_addr=bind_addr))
        finally:
            self._loop.run_until_complete(*self._cleanup)
            self._loop.close()

    async def serve(self, bind_addr: str) -> None:
        self.add_insecure_port(bind_addr)

        await self.startup()

        self._cleanup.append(self.shutdown())

        await self.wait_for_termination()

    async def startup(self) -> None:
        # Running on_startup callback.
        for handler in self.on_startup:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

        await self.server.start()

    async def shutdown(self):
        # Running on_startup callback.
        for handler in self.on_shutdown:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

        await self.server.stop(grace=self._grace_period)

    async def wait_for_termination(self, timeout: int | None = None) -> bool:
        return await self.server.wait_for_termination(timeout=timeout)

    def add_insecure_port(self, address: str) -> int:
        return self.server.add_insecure_port(address)

    def add_secure_port(self, address: str, credentials: grpc.ServerCredentials) -> int:
        return self.server.add_secure_port(address, credentials)
