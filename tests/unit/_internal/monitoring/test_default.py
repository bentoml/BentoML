from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from bentoml._internal.monitoring.default import DefaultMonitor

ROTATE_MONITOR_LOGS = """
import logging
import bentoml

with bentoml.monitor("retention") as monitor:
    monitor.log("initial", "value", "feature", "categorical")

handler = monitor.data_logger.handlers[0]
for index in range(4):
    handler.rolloverAt = 1700000000 + (index + 1) * handler.interval
    handler.doRollover()
    monitor.data_logger.info({"value": index})

logging.shutdown()
"""


@pytest.mark.parametrize(
    "backup_count, custom_backup_count, expected_files",
    [(None, None, 4), (0, None, 4), (2, None, 2), (1, 3, 3)],
)
def test_default_monitor_log_retention(
    tmp_path: Path,
    backup_count: int | None,
    custom_backup_count: int | None,
    expected_files: int,
) -> None:
    env = os.environ.copy()
    env.pop("BENTOML_CONFIG", None)
    env.pop("BENTOML_CONFIG_OPTIONS", None)
    env.pop("BENTOML_CONFIG_OVERRIDES", None)
    options: dict[str, Any] = {}
    if backup_count is not None:
        options["log_backup_count"] = backup_count
    if custom_backup_count is not None:
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "data": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "filename": "{data_filename}",
                    "when": "D",
                    "backupCount": custom_backup_count,
                    "formatter": "json",
                }
            },
            "formatters": {
                "json": {"class": "pythonjsonlogger.jsonlogger.JsonFormatter"}
            },
            "loggers": {
                "bentoml_monitor_data": {
                    "level": "INFO",
                    "handlers": ["data"],
                    "propagate": False,
                }
            },
        }
        (tmp_path / "logging.yaml").write_text(yaml.safe_dump(config))
        options["log_config_file"] = "logging.yaml"
    if options:
        env["BENTOML_CONFIG_OVERRIDES"] = json.dumps(
            {"monitoring": {"options": options}}
        )
    result = subprocess.run(
        [sys.executable, "-c", ROTATE_MONITOR_LOGS],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data_path = tmp_path / "monitoring" / "retention" / "data"
    backups = sorted(data_path.glob("data.0.log.*"))
    assert len(backups) == expected_files
    assert [json.loads(path.read_text())["value"] for path in backups] == [
        "initial",
        0,
        1,
        2,
    ][-expected_files:]
    assert json.loads((data_path / "data.0.log").read_text())["value"] == 3
    if custom_backup_count is None:
        assert (
            tmp_path / "monitoring" / "retention" / "schema" / "schema.0.log"
        ).is_file()


@pytest.mark.parametrize("backup_count", [-1, 1.5, "2", True, False])
def test_default_monitor_rejects_invalid_backup_count(
    tmp_path: Path, backup_count: Any
) -> None:
    with pytest.raises(
        ValueError, match="log_backup_count must be a non-negative integer"
    ):
        DefaultMonitor("retention", str(tmp_path), log_backup_count=backup_count)
