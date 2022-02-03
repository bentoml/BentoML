import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circus.arbiter import Arbiter  # type: ignore[reportMissingTypeStubs]
    from circus.watcher import Watcher  # type: ignore[reportMissingTypeStubs]


def create_standalone_arbiter(
    watchers: t.List["Watcher"], **kwargs: t.Any
) -> "Arbiter":
    from circus.arbiter import Arbiter  # type: ignore[reportMissingTypeStubs]

    from . import reserve_free_port

    with reserve_free_port() as endpoint_port:
        with reserve_free_port() as pubsub_port:
            return Arbiter(
                watchers,
                endpoint=f"tcp://127.0.0.1:{endpoint_port}",
                pubsub_endpoint=f"tcp://127.0.0.1:{pubsub_port}",
                **kwargs,
            )
