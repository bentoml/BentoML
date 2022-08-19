from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from grpc import aio


class Config:
    @inject
    def __init__(
        self,
        bind_address: str,
        enable_reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
        max_message_length: int
        | None = Provide[BentoMLContainer.grpc.max_message_length],
        max_concurrent_streams: int
        | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
        maximum_concurrent_rpcs: int
        | None = Provide[BentoMLContainer.grpc.maximum_concurrent_rpcs],
        migration_thread_pool_workers: int = 1,
        graceful_shutdown_timeout: float = 5.0,
    ) -> None:

        # Note that the max_workers are used inside ThreadPoolExecutor.
        # This ThreadPoolExecutor are used by aio.Server() to execute non-AsyncIO RPC handlers.
        # Setting it to 1 makes it thread-safe for sync APIs.
        self.migration_thread_pool_workers = migration_thread_pool_workers

        # maximum_concurrent_rpcs defines the maximum number of concurrent RPCs this server
        # will service before returning RESOURCE_EXHAUSTED status.
        # Set to None will indicate no limit.
        self.maximum_concurrent_rpcs = maximum_concurrent_rpcs

        self.bind_address = bind_address
        self.enable_reflection = enable_reflection
        self.graceful_shutdown_timeout = graceful_shutdown_timeout

        options: aio.ChannelArgumentType = []

        if sys.platform != "win32":
            # https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/grpc_types.h#L294
            # Eventhough GRPC_ARG_ALLOW_REUSEPORT is set to 1 by default, we want still
            # want to explicitly set it to 1 so that we can spawn multiple gRPC servers in
            # production settings.
            options.append(("grpc.so_reuseport", 1))

        if max_concurrent_streams:
            options.append(("grpc.max_concurrent_streams", max_concurrent_streams))

        if max_message_length:
            options.extend(
                (
                    ("grpc.max_message_length", max_message_length),
                    ("grpc.max_receive_message_length", max_message_length),
                    ("grpc.max_send_message_length", max_message_length),
                    ("grpc.max_send_message_length", max_message_length),
                )
            )

        self.options = tuple(options)
