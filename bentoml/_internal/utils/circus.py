from __future__ import annotations

import typing as t
import logging
import pathlib
import threading
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import fs
import watchfiles
from pathspec import PathSpec
from simple_di import inject
from simple_di import Provide
from circus.plugins import CircusPlugin

from ..configuration import is_pypi_installed_bentoml
from ..bento.build_config import BentoBuildConfig
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from circus.arbiter import Arbiter
    from circus.sockets import CircusSocket
    from circus.watcher import Watcher
    from watchfiles.main import FileChange

    from ..types import PathType

logger = logging.getLogger(__name__)


def create_circus_socket_from_uri(
    uri: str,
    *args: t.Any,
    name: str = "",
    **kwargs: t.Any,
) -> CircusSocket | t.NoReturn:
    from circus.sockets import CircusSocket

    from ..utils.uri import uri_to_path

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

    from . import reserve_free_port

    with reserve_free_port() as endpoint_port:
        with reserve_free_port() as pubsub_port:
            return Arbiter(
                watchers,
                endpoint=f"tcp://127.0.0.1:{endpoint_port}",
                pubsub_endpoint=f"tcp://127.0.0.1:{pubsub_port}",
                **kwargs,
            )


class ReloaderPlugin(CircusPlugin):
    """
    A circus plugin that reloads the BentoService when the service code changes.


    Args:
        working_dir: The directory of the bento service.
        reload_delay: The delay in seconds between checking for changes.
    """

    name = "bento_change_reloader"
    config: dict[str, t.Any]

    @inject
    def __init__(self, *args: t.Any, **config: t.Any):
        assert "bento_identifier" in config, "`bento_identifier` is required"
        assert "working_dir" in config, "`working_dir` is required"

        super().__init__(*args, **config)  # type: ignore (unfinished types for circus)

        self.name = self.config.get("name")
        working_dir: str = self.config["working_dir"]

        # circus/plugins/__init__.py:282 -> converts all given configs to dict[str, str]
        self.reload_delay: float = float(self.config.get("reload_delay", 1))
        self.reloader = FileWatcher(working_dir=working_dir)

    def look_after(self):
        while True:
            try:
                changes = next(self.reloader)
                if changes not in (None, []):
                    logger.warning(
                        "%s detected changes in %s. Reloading...",
                        self.__class__.__name__,
                        ", ".join(map(display_path, changes or [])),
                    )
                    raise StopIteration
            except StopIteration:
                self.call("restart", name="*")  # type: ignore
                break

    def handle_init(self):
        from tornado.ioloop import PeriodicCallback

        self.period = PeriodicCallback(self.look_after, self.reload_delay * 1000)
        self.period.start()

    def handle_stop(self):
        self.period.stop()

    def handle_recv(self, data: t.Any):
        pass


class FileWatcher:
    def __init__(
        self,
        working_dir: str,
        *,
        bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
        steps: int = 0,  # we want to make the notifier not blocking
        rust_timeout: int = 0,  # we don't wan to set timeout, leverage PeriodicCallback
    ):
        watch_dirs = [working_dir, fs.path.combine(bentoml_home, "models")]

        if not is_pypi_installed_bentoml():
            # git root from this __file__
            git_root = str(pathlib.Path(__file__).parent.parent.parent.parent)
            watch_dirs.append(git_root)

        logger.info(f"Watching directories: {watch_dirs}")

        self.exit_event = threading.Event()

        self.filters = BentoFilter(pathlib.Path.cwd())
        self.watchers = watchfiles.watch(
            *watch_dirs,
            watch_filter=None,
            step=steps,
            rust_timeout=rust_timeout,
            stop_event=self.exit_event,
            yield_on_timeout=True,
        )

    def __iter__(self) -> t.Iterator[list[pathlib.Path] | None]:
        return self

    def __next__(self) -> list[pathlib.Path] | None:
        return self.should_restart()

    def should_restart(self) -> list[pathlib.Path] | None:
        changes = next(self.watchers)
        if changes:
            self.exit_event.set()
            return self.filtered(changes)
        else:
            raise StopIteration

    def filtered(self, changes: t.Set[FileChange]) -> list[pathlib.Path] | None:
        unique_paths = {pathlib.Path(c[1]) for c in changes}
        logger.warning([p for p in unique_paths if self.filters(p.__fspath__())])
        return [p for p in unique_paths if self.filters(p.__fspath__())]


class BentoFilter:
    """
    A Python filter that respect .bentoignore
    Currently ignoring `bentofile.exclude` fields.
    """

    def __init__(self, path: PathType):
        if isinstance(path, pathlib.Path):
            path = path.__fspath__()
        else:
            path = str(path)

        self._fs = fs.open_fs(path)

        if self._fs.exists("bentofile.yaml"):
            with self._fs.open("bentofile.yaml", "r", encoding="utf-8") as f:
                build_config = BentoBuildConfig.from_yaml(f.read()).with_defaults()
                include = build_config.include
                exclude = build_config.exclude
        else:
            include = ["*"]
            exclude = []

        # bentoml/_internal/bento/bento.py#L191
        self.include_spec = PathSpec.from_lines("gitwildmatch", include)
        self.exclude_spec = PathSpec.from_lines("gitwildmatch", exclude)

    def __call__(self, path: str) -> bool:
        bentoignore_spec: list[tuple[str, PathSpec]] = []
        for dir_path, _, files in self._fs.walk():
            for ignore_file in [f for f in files if f.name == ".bentoignore"]:
                bentoignore_spec.append(
                    (
                        dir_path,
                        PathSpec.from_lines(
                            "gitwildmatch",
                            self._fs.open(ignore_file.make_path(dir_path)),
                        ),
                    )
                )

        cur_exclude_specs: list[tuple[str, PathSpec]] = []
        for ignore_path, spec in bentoignore_spec:
            if fs.path.isparent(ignore_path, fs.path.dirname(path)):
                cur_exclude_specs.append((ignore_path, spec))

        logger.warning(cur_exclude_specs)
        if self.include_spec.match_file(path) and not self.exclude_spec.match_file(
            path
        ):
            if not any(
                spec.match_file(fs.path.relativefrom(ignore_path, path))
                for ignore_path, spec in cur_exclude_specs
            ):
                return True
        return False


def display_path(path: pathlib.Path) -> str:
    try:
        return f"'{path.relative_to(pathlib.Path.cwd())}'"
    except ValueError:
        return f"'{path}'"
