from __future__ import annotations

import typing as t
import logging
import pathlib
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from circus.plugins import CircusPlugin  # type: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from circus.arbiter import Arbiter  # type: ignore[reportMissingTypeStubs]
    from circus.sockets import CircusSocket  # type: ignore[reportMissingTypeStubs]
    from circus.watcher import Watcher  # type: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)


def create_circus_socket_from_uri(
    uri: str,
    *args: t.Any,
    name: str = "",
    **kwargs: t.Any,
) -> CircusSocket:
    from circus.sockets import CircusSocket  # type: ignore[reportMissingTypeStubs]

    from bentoml._internal.utils.uri import uri_to_path

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


# TODO: use svc.build_args.include/exclude as default files to watch
# TODO: watch changes in model store when "latest" model tag is used


class BentoChangeReloader(CircusPlugin):
    """
    A circus plugin that reloads the BentoService when the service code changes.


    Args:
        working_dir: The directory of the bento service.
        reload_delay: The delay in seconds between checking for changes.
    """

    name = "bento_change_reloader"
    config: dict[str, t.Any]

    def __init__(self, *args: t.Any, **config: t.Any):
        assert "bento_identifier" in config, "`bento_identifier` is required"
        assert "working_dir" in config, "`working_dir` is required"

        super().__init__(*args, **config)  # type: ignore (unfinished types for circus)

        self.name = self.config.get("name")
        working_dir: str = self.config["working_dir"]

        # circus/plugins/__init__.py:282 -> converts all given configs to dict[str, str]
        self.reload_delay: float = float(self.config.get("reload_delay", 1))
        self.file_watcher = PyFileChangeWatcher([working_dir])

    def look_after(self):
        if self.file_watcher.is_file_changed():
            logger.info("Restarting...")
            self.call("restart", name="*")  # type: ignore
            self.file_watcher.reset()

    def handle_init(self):
        from tornado import ioloop

        self.period = ioloop.PeriodicCallback(self.look_after, self.reload_delay * 1000)
        self.period.start()

    def handle_stop(self):
        self.period.stop()

    def handle_recv(self, data: t.Any):
        pass


class PyFileChangeWatcher:
    def __init__(
        self,
        watch_dirs: list[pathlib.Path] | list[str] | None = None,
    ) -> None:
        self.mtimes: dict[pathlib.Path, float] = {}
        if not watch_dirs:
            watch_dirs = [pathlib.Path.cwd()]
        self.watch_dirs = [
            pathlib.Path(d) if isinstance(d, str) else d for d in watch_dirs
        ]
        logger.info(f"Watching directories: {', '.join(map(str, self.watch_dirs))}")

    def is_file_changed(self) -> bool:
        for file in self.iter_files():
            try:
                mtime = file.stat().st_mtime
            except OSError:  # pragma: nocover
                continue

            old_time = self.mtimes.get(file)
            if old_time is None:
                self.mtimes[file] = mtime
                continue
            elif mtime > old_time:
                display_path = str(file)
                try:
                    display_path = str(file.relative_to(pathlib.Path.cwd()))
                except ValueError:
                    pass
                message = "Detected file change in '%s'"
                logger.warning(message, display_path)
                return True
        return False

    def reset(self) -> None:
        self.mtimes = {}

    def iter_files(self) -> t.Iterator[pathlib.Path]:
        for reload_dir in self.watch_dirs:
            for path in list(reload_dir.rglob("*.py")):
                yield path.resolve()
