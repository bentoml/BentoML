from __future__ import annotations

import contextlib
import typing as t
from threading import Thread
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import attrs
from circus.arbiter import Arbiter as _Arbiter

if TYPE_CHECKING:
    from circus.sockets import CircusSocket
    from circus.watcher import Watcher

__all__ = [
    "create_circus_socket_from_uri",
    "create_standalone_arbiter",
]


class Arbiter(_Arbiter):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.exit_stack = contextlib.ExitStack()

    def start(self, cb: t.Callable[[t.Any], t.Any] | None = None) -> None:
        self.exit_stack.__enter__()
        fut = super().start(cb)
        if exc := fut.exception():
            raise exc

    def stop(self) -> None:
        self.exit_stack.__exit__(None, None, None)
        return super().stop()


class ThreadedArbiter(Arbiter):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._thread: Thread | None = None

    def start(self, cb: t.Callable[[t.Any], t.Any] | None = None) -> None:
        self._thread = Thread(target=self._worker, args=(cb,), daemon=True)
        self._thread.start()

    def _worker(self, cb: t.Callable[[t.Any], t.Any] | None = None) -> None:
        # reset the loop in thread
        self.loop = None
        self._ensure_ioloop()
        self.ctrl.loop = self.loop  # type: ignore[union-attr]
        super().start(cb)

    def stop(self) -> None:
        if self.loop is not None:
            self.loop.add_callback(super().stop)  # type: ignore[union-attr]
        if self._thread is not None:
            self._thread.join()


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


def create_standalone_arbiter(
    watchers: list[Watcher], *, threaded: bool = False, **kwargs: t.Any
) -> Arbiter:
    from .. import reserve_free_port

    arbiter_cls = ThreadedArbiter if threaded else Arbiter

    with reserve_free_port() as endpoint_port:
        with reserve_free_port() as pubsub_port:
            return arbiter_cls(
                watchers,
                endpoint=f"tcp://127.0.0.1:{endpoint_port}",
                pubsub_endpoint=f"tcp://127.0.0.1:{pubsub_port}",
                # XXX: Currently ,the default check_delay will always raise ConflictError. This probably has to do with
                # the runners server is not ready in time when the controller run healthcheck.
                check_delay=kwargs.pop("check_delay", 10),
                **kwargs,
            )


@attrs.frozen
class Server:
    url: str
    arbiter: Arbiter = attrs.field(repr=False)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.arbiter.stop()

    @property
    def running(self) -> bool:
        return self.arbiter.running

    def __enter__(self) -> Server:
        return self

    def __exit__(self, exc_type: t.Any, exc_value: t.Any, traceback: t.Any) -> None:
        self.stop()
