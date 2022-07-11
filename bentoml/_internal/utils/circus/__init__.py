from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from circus.arbiter import Arbiter
    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

__all__ = [
    "create_circus_socket_from_uri",
    "create_standalone_arbiter",
]


def create_circus_socket_from_uri(
    uri: str, *args: t.Any, name: str = "", **kwargs: t.Any
) -> CircusSocket:
    from circus.sockets import CircusSocket

    from ..uri import uri_to_path

    parsed = urlparse(uri)
    if parsed.scheme in ("file", "unix"):
        return CircusSocket(
            name=name,
            path=uri_to_path(uri),
            *args,
            **kwargs,
        )
    elif parsed.scheme == "tcp":
        return CircusSocket(
            name=name,
            host=parsed.hostname,
            port=parsed.port,
            *args,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")


def create_standalone_arbiter(watchers: list[Watcher], **kwargs: t.Any) -> Arbiter:
    from circus.arbiter import Arbiter

    from .. import reserve_free_port

    with reserve_free_port() as endpoint_port:
        with reserve_free_port() as pubsub_port:
            return Arbiter(
                watchers,
                endpoint=f"tcp://127.0.0.1:{endpoint_port}",
                pubsub_endpoint=f"tcp://127.0.0.1:{pubsub_port}",
                **kwargs,
            )
