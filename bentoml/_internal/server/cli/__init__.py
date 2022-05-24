from __future__ import annotations

import typing
import logging
import pathlib

logger = logging.getLogger(__name__)


from tornado import ioloop
from circus.plugins import CircusPlugin

if typing.TYPE_CHECKING:
    pass


class BentoChangeReloader(CircusPlugin):

    name = "command_reloader"

    def __init__(
        self,
        *args: typing.Any,
        watch_dirs: list[pathlib.Path] | None = None,
        loop_rate: int = 1,
        **config: typing.Any,
    ):
        super().__init__(*args, **config)
        self.name = config.get("name")
        self.loop_rate = loop_rate
        self.cmd_files = {}
        self.period: ioloop.PeriodicCallback
        self.file_watcher = PyFileChangeWatcher(watch_dirs)

    def look_after(self):
        if self.file_watcher.is_file_changed():
            logger.info("file modified. Restarting.")
            self.call("restart")
            self.file_watcher.reset()

    def handle_init(self):
        self.period = ioloop.PeriodicCallback(self.look_after, self.loop_rate * 1000)
        self.period.start()

    def handle_stop(self):
        self.period.stop()

    def handle_recv(self, data: typing.Any):
        pass


class PyFileChangeWatcher:
    def __init__(
        self,
        watch_dirs: list[pathlib.Path] | None = None,
    ) -> None:
        self.mtimes: dict[pathlib.Path, float] = {}
        if not watch_dirs:
            watch_dirs = [pathlib.Path.cwd()]
        self.watch_dirs = watch_dirs

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

    def iter_files(self) -> typing.Iterator[pathlib.Path]:
        for reload_dir in self.watch_dirs:
            for path in list(reload_dir.rglob("*.py")):
                yield path.resolve()
