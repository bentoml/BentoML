from __future__ import annotations

import os
import typing as t
import logging
from abc import abstractmethod
from pathlib import Path
from functools import lru_cache

import fs
from circus.plugins import CircusPlugin

from bentoml.exceptions import BentoMLException

from ...utils.pkg import source_locations
from ...configuration import is_pypi_installed_bentoml
from ...bento.build_config import BentoBuildConfig
from ...bento.build_config import BentoPatternSpec

logger = logging.getLogger(__name__)


class ReloaderPlugin(CircusPlugin):
    """
    A Circus plugin that reloads the BentoService whenever the service code changes.

    A child class should implement the following methods:
        - has_modification()
        - handle_stop()
        - property watcher()

    An optional '_post_restart' hook can be implemented. Note that this will only be called after
    'self.call("restart", name="*")'.

    This plugin provides `file_changed(path)` to detect whether a file has changed.
    It will respect the 'include' and 'exclude' in bentofile.yaml as well as '.bentoignore'.
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

        self.create_spec(_internal=True)

    @lru_cache(maxsize=1)
    def create_spec(self, *, _internal: bool = False) -> None:
        if not _internal:
            raise BentoMLException(
                "'create_spec()' is internal function and should not used directly."
            )
        bentofile_path = os.path.join(self.working_dir, "bentofile.yaml")
        if not os.path.exists(bentofile_path):
            # if bentofile.yaml is not found, by default we will assume to watch all files
            # via BentoBuildConfig.with_defaults()
            build_config = BentoBuildConfig(service="").with_defaults()
        else:
            # respect bentofile.yaml include and exclude
            with open(bentofile_path, "r") as f:
                build_config = BentoBuildConfig.from_yaml(f).with_defaults()
        self.build_config = build_config

        self.bento_spec = BentoPatternSpec(build_config)

    def file_changed(self, path: str | Path) -> bool:
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
