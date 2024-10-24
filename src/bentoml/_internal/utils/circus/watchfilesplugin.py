from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from threading import Event
from threading import Thread
from typing import TYPE_CHECKING

import fs
from circus.plugins import CircusPlugin
from watchfiles import watch

from ...bento.build_config import BentoBuildConfig
from ...bento.build_config import BentoPathSpec
from ...configuration import is_editable_bentoml
from ...context import server_context
from ...log import configure_server_logging
from ...utils.pkg import source_locations

if TYPE_CHECKING:
    from watchfiles.main import FileChange


logger = logging.getLogger(__name__)


class ServiceReloaderPlugin(CircusPlugin):
    name = "service_reloader"
    config: dict[str, str]

    def __init__(self, *args: t.Any, **config: t.Any):
        assert "working_dir" in config, "`working_dir` is required"

        configure_server_logging()
        server_context.service_type = "observer"

        super().__init__(*args, **config)

        # circus/plugins/__init__.py:282 -> converts all given configs to dict[str, str]
        self.name = self.config.get("name")
        self.working_dir = self.config["working_dir"]

        # a list of folders to watch for changes
        watch_dirs = [self.working_dir]
        self._specs = [(Path(self.working_dir).as_posix(), self.create_spec())]

        if is_editable_bentoml():
            # bentoml src from this __file__
            bentoml_src = Path(source_locations("bentoml")).parent
            logger.info(
                "BentoML is installed via development mode, adding source root to 'watch_dirs'."
            )
            watch_dirs.append(str(bentoml_src))
            self._specs.append(
                (
                    bentoml_src.as_posix(),
                    BentoPathSpec(
                        # only watch python files in bentoml src
                        ["*.py", "*.yaml"],
                        [],
                        bentoml_src.as_posix(),
                        recurse_ignore_filename=".gitignore",
                    ),
                )
            )

        logger.info("Watching directories: %s", watch_dirs)

        # a thread to restart circus
        self.exit_event = Event()

        self.file_changes = watch(
            *watch_dirs,
            watch_filter=None,
            yield_on_timeout=True,  # stop hanging on our tests, doesn't affect the behaviour
            stop_event=self.exit_event,
            rust_timeout=0,  # we should set timeout to zero for no timeout
        )

    def create_spec(self) -> BentoPathSpec:
        build_config = BentoBuildConfig.from_bento_dir(self.working_dir)
        return BentoPathSpec(
            build_config.include, build_config.exclude, self.working_dir
        )

    def should_include(self, path: Path) -> bool:
        # returns True if file with 'path' has changed, else False
        str_path = path.as_posix()
        for parent, spec in self._specs:
            if fs.path.isparent(parent, str_path):
                return spec.includes(fs.path.relativefrom(parent, str_path))
        return False

    def has_modification(self) -> bool:
        for changes in self.file_changes:
            uniq_paths = {Path(c[1]) for c in changes}
            filtered = [c for c in uniq_paths if self.should_include(c)]
            if filtered:
                change_type, path = self.display_path(changes)
                logger.warning("%s: %s", change_type.upper(), path)
                return True
        return False

    def look_after(self):
        if self.has_modification():
            logger.warning("Detected changes. Reloading...")
            self.call("restart", name="*")

    def handle_init(self):
        self.watcher = Thread(target=self.look_after)
        self.watcher.start()

    def handle_recv(self, data: t.Any):
        pass

    def handle_stop(self) -> None:
        self.watcher.join()
        self.exit_event.set()

    def display_path(self, change: set[FileChange]) -> tuple[str, str]:
        type_change, path = change.pop()
        try:
            return (
                type_change.raw_str(),
                f"'{Path(path).relative_to(self.working_dir)}'",
            )
        except ValueError:
            return type_change.raw_str(), f"'{path}'"
