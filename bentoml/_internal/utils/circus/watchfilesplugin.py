from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from threading import Event
from threading import Thread

from bentoml.exceptions import MissingDependencyException

from .baseplugin import ReloaderPlugin

if TYPE_CHECKING:
    from watchfiles.main import FileChange

try:
    from watchfiles import watch
except ImportError:
    raise MissingDependencyException(
        "'watchfiles' is required to use with '--reload-backend=watchfiles'. Install watchfiles with `pip install 'bentoml[watchfiles]'`"
    )

logger = logging.getLogger(__name__)


class WatchFilesPlugin(ReloaderPlugin):
    def __init__(self, *args: t.Any, **config: t.Any):
        super().__init__(*args, **config)

        # NOTE: we ignore --reload-delay for watchfiles as we rely on 'rust_timeout'
        logger.debug("'--reload-delay' will be ignored when using 'watchfiles'.")
        self.reload_delay = 0

        # a thread to restart circus
        self.should_exit = Event()

        self.file_changes = watch(
            *self.watch_dirs,
            watch_filter=None,
            yield_on_timeout=True,  # NOTE: stop hanging on our tests, doesn't affect the behaviour
            stop_event=self.should_exit,
            rust_timeout=self.reload_delay,
        )

    def has_modification(self) -> bool:
        for changes in self.file_changes:
            uniq_paths = {Path(c[1]) for c in changes}
            filtered = [c for c in uniq_paths if self.should_include(c)]
            if filtered:
                change_type, path = self._display_path(changes)
                logger.warning(f"{change_type.upper()}: {path}")
                return True
        return False

    @property
    def watcher(self) -> Thread:
        return Thread(target=self.look_after)

    def handle_stop(self) -> None:
        self.watcher.join()
        self.should_exit.set()

    def _display_path(self, change: set[FileChange]) -> tuple[str, str]:
        type_change, path = change.pop()
        try:
            return (
                type_change.raw_str(),
                f"'{Path(path).relative_to(self.working_dir)}'",
            )
        except ValueError:
            return type_change.raw_str(), f"'{path}'"
