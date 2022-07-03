from __future__ import annotations

import os
import typing as t
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
from pathlib import Path

import fs
from circus.plugins import CircusPlugin

from bentoml.exceptions import BentoMLException

from ...utils.pkg import source_locations
from ...configuration import is_pypi_installed_bentoml
from ...bento.build_config import IgnoreSpec
from ...bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from fs.base import FS

logger = logging.getLogger(__name__)


class ReloaderPlugin(CircusPlugin):
    """
    A circus plugin that reloads the BentoService when the service code changes.

    A child class should implement the following methods:
        - has_modification()
        - handle_stop()
        - _post_restart() (optional)

    A child class should also implement the following property:
        - watcher

    Note that _post_restart() is only called after restart if file changes are detected.
    """

    name = "bento_file_watcher"
    config: dict[str, str]

    def __init__(self, *args: t.Any, **config: t.Any):
        assert "working_dir" in config, "`working_dir` is required"

        super().__init__(*args, **config)

        # circus/plugins/__init__.py:282 -> converts all given configs to dict[str, str]
        self.name = self.config.get("name")
        self.bentoml_home = self.config["bentoml_home"]
        self.working_dir = self.config["working_dir"]
        self.reload_delay = float(self.config.get("reload_delay", 1))

        # a list of folders to watch for changes
        watch_dirs = [self.working_dir, os.path.join(self.bentoml_home, "models")]

        if not is_pypi_installed_bentoml():
            # bentoml src from this __file__
            logger.info(
                "BentoML is installed via development mode, adding source root to 'watch_dirs'"
            )
            watch_dirs.append(t.cast(str, source_locations("bentoml")))

        logger.info(f"Watching directories: {watch_dirs}")
        self.watch_dirs = watch_dirs
        self.watch_fs = [fs.open_fs(d) for d in self.watch_dirs]

    def file_changed(self, path: str | Path) -> bool:
        # returns True if file with 'path' has changed, else False
        if isinstance(path, Path):
            path = path.__fspath__()

        bentofile_path = os.path.join(self.working_dir, "bentofile.yaml")
        if not os.path.exists(bentofile_path):
            # if bentofile.yaml is not found, by default we will assume to watch all files
            # via BentoBuildConfig.with_defaults()
            build_config = BentoBuildConfig(service="").with_defaults()
        else:
            # respect bentofile.yaml include and exclude
            with open(bentofile_path, "r") as f:
                build_config = BentoBuildConfig.from_yaml(f).with_defaults()

        return any(
            IgnoreSpec(build_config).includes(
                path, current_dir=os.path.dirname(path), ctx_fs=ctx_fs
            )
            for ctx_fs in self.watch_fs
        )

    @abstractmethod
    def has_modification(self) -> bool:
        # returns True if file changed, False otherwise
        raise NotImplementedError("'is_modified()' is not implemented.")

    @property
    @abstractmethod
    def watcher(self) -> t.Any:
        raise NotImplementedError("'watcher()' is not implemented.")

    def _post_restart(self):
        # _post_restart callback is only called after restart if file changes are detected.
        # this hook is optional.
        pass

    def look_after(self):
        if self.has_modification():
            logger.warning(f"{self.__class__.__name__} detected changes. Reloading...")
            self.call("restart", name="*")
            if hasattr(self, "_post_restart"):
                self._post_restart()

    def handle_init(self):
        if not hasattr(self.watcher, "start"):
            raise BentoMLException(
                f"'start()' callback is required for {self.watcher.__class__.__name__}."
            )
        self.watcher.start()

    @abstractmethod
    def handle_stop(self) -> None:
        raise NotImplementedError("'handle_stop()' is not implemented.")

    def handle_recv(self, data: t.Any):
        pass
