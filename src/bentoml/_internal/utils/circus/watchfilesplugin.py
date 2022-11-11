from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from threading import Event
from threading import Thread

import fs
from watchfiles import watch
from circus.plugins import CircusPlugin

from ...log import configure_server_logging
from ...context import component_context
from ...utils.pkg import source_locations
from ...configuration import is_pypi_installed_bentoml
from ...bento.build_config import BentoPathSpec
from ...bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from watchfiles.main import FileChange


logger = logging.getLogger(__name__)


class ServiceReloaderPlugin(CircusPlugin):
    name = "service_reloader"
    config: dict[str, str]

    def __init__(self, *args: t.Any, **config: t.Any):
        assert "working_dir" in config, "`working_dir` is required"

        configure_server_logging()
        component_context.component_type = "observer"

        super().__init__(*args, **config)

        # circus/plugins/__init__.py:282 -> converts all given configs to dict[str, str]
        self.name = self.config.get("name")
        self.bentoml_home = self.config["bentoml_home"]
        self.working_dir = self.config["working_dir"]

        # a list of folders to watch for changes
        watch_dirs = [self.working_dir, os.path.join(self.bentoml_home, "models")]

        if not is_pypi_installed_bentoml():
            # bentoml src from this __file__
            logger.info(
                "BentoML is installed via development mode, adding source root to 'watch_dirs'."
            )
            watch_dirs.append(t.cast(str, source_locations("bentoml")))

        logger.info("Watching directories: %s", watch_dirs)
        self.watch_dirs = watch_dirs

        self.create_spec()

        # a thread to restart circus
        self.exit_event = Event()

        self.file_changes = watch(
            *self.watch_dirs,
            watch_filter=None,
            yield_on_timeout=True,  # stop hanging on our tests, doesn't affect the behaviour
            stop_event=self.exit_event,
            rust_timeout=0,  # we should set timeout to zero for no timeout
        )

    def create_spec(self) -> None:
        bentofile_path = os.path.join(self.working_dir, "bentofile.yaml")
        if not os.path.exists(bentofile_path):
            # if bentofile.yaml is not found, by default we will assume to watch all files
            # via BentoBuildConfig.with_defaults()
            build_config = BentoBuildConfig(service="").with_defaults()
        else:
            # respect bentofile.yaml include and exclude
            with open(bentofile_path, "r") as f:
                build_config = BentoBuildConfig.from_yaml(f).with_defaults()

        self.bento_spec = BentoPathSpec(build_config.include, build_config.exclude)  # type: ignore (unfinished converter type)

    def should_include(self, path: str | Path) -> bool:
        # returns True if file with 'path' has changed, else False
        if isinstance(path, Path):
            path = path.__fspath__()

        return any(
            self.bento_spec.includes(
                path,
                recurse_exclude_spec=filter(
                    lambda s: fs.path.isparent(s[0], os.path.dirname(path)),
                    self.bento_spec.from_path(dirs),
                ),
            )
            for dirs in self.watch_dirs
        )

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
