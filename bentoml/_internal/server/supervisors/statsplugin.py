from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path

from .baseplugin import ReloaderPlugin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tornado.ioloop import PeriodicCallback


class StatsPlugin(ReloaderPlugin):
    def __init__(self, *args: t.Any, **config: t.Any):
        super().__init__(*args, **config)
        self.mtimes: dict[Path, float] = {}

    def _post_restart(self) -> None:
        self.mtimes = {}

    @property
    def watcher(self) -> PeriodicCallback:
        from tornado import ioloop

        return ioloop.PeriodicCallback(self.look_after, self.reload_delay * 1000)

    def is_modified(self) -> bool:
        for file in self.file_changes():
            try:
                mtime = file.stat().st_mtime
            except OSError:  # pragma: no cover
                continue

            old_time = self.mtimes.get(file)
            if old_time is None:
                self.mtimes[file] = mtime
                continue
            elif mtime > old_time:
                display_path = str(file)
                try:
                    display_path = file.relative_to(self.working_dir).__fspath__()
                except ValueError:
                    pass
                message = "Detected file change in '%s'"
                logger.warning(message, display_path)
                return True
        return False

    def handle_stop(self) -> None:
        self.watcher.stop()

    def file_changes(self) -> t.Generator[Path, None, None]:
        yield from [Path(dir_, p) for dir_, p in self.watch_files()]
