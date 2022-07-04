from __future__ import annotations

import os
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from unittest.mock import patch

import pytest
from circus.tests.support import TestCircus

from bentoml._internal.utils.circus.statsplugin import StatsPlugin

if TYPE_CHECKING:
    from unittest.mock import MagicMock


@pytest.mark.usefixtures("reload_directory")
class TestStatsPlugin(TestCircus):
    reload_directory: Path

    def setUp(self) -> None:
        super().setUp()
        self.plugin_kwargs: dict[str, t.Any] = {
            "bento_identifier": ".",
            "working_dir": self.reload_directory.__fspath__(),
            "bentoml_home": self.reload_directory.__fspath__(),
        }

    def test_logging_info(self) -> None:
        with self.assertLogs("bentoml", level=logging.INFO) as log:
            self.make_plugin(StatsPlugin, **self.plugin_kwargs)
            self.assertIn("adding source root", log.output[0])
            self.assertIn("Watching directories", log.output[1])

    def test_reloader_params_is_required(self) -> None:
        self.assertRaises(AssertionError, self.make_plugin, StatsPlugin)  # type: ignore (unfinished circus type)

    def test_default_timeout(self) -> None:
        plugin = self.make_plugin(StatsPlugin, **self.plugin_kwargs)
        self.assertEqual(plugin.reload_delay, 1.0)

    def setup_call_mock(self, watcher_name: str) -> MagicMock:
        patcher = patch.object(StatsPlugin, "call")
        call_mock = patcher.start()
        self.addCleanup(patcher.stop)
        call_mock.side_effect = [
            {"watchers": [watcher_name]},
            {"options": {"cmd": watcher_name}},
            None,
        ]
        return call_mock

    def test_look_after_trigger_restart(self) -> None:
        file = self.reload_directory.joinpath("file.txt").__fspath__()

        call_mock = self.setup_call_mock(watcher_name="reloader")
        plugin = self.make_plugin(StatsPlugin, **self.plugin_kwargs)
        Path(file).touch()

        with self.assertLogs("bentoml", level=logging.WARNING) as log:
            with Path(file).open("w") as f:
                f.write("xff")
            plugin.look_after()
            call_mock.assert_called_with("restart", name="*")
            self.assertIn("Detected change", log.output[0])

    def test_look_after_trigger_restart_on_deletion(self):
        file = self.reload_directory.joinpath("train.py").__fspath__()

        call_mock = self.setup_call_mock(watcher_name="reloader")
        plugin = self.make_plugin(StatsPlugin, **self.plugin_kwargs)
        os.remove(file)

        with self.assertLogs("bentoml", level=logging.WARNING) as log:
            plugin.look_after()
            call_mock.assert_called_with("restart", name="*")
            self.assertIn("Detected change", log.output[0])
