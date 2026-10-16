from __future__ import annotations

import json
import subprocess
import sys
import typing as t
from pathlib import Path

import pytest

LOGGER_NAMES = ("bentoml_monitor_data", "bentoml_monitor_schema")


def run_public_monitor(
    tmp_path: Path,
    config_text: str | None,
    *,
    suffix: str = "yaml",
    monitor_name: str = "brace_probe",
) -> dict[str, t.Any]:
    config_file = tmp_path / f"logging.{suffix}"  # type: ignore[operator]
    if config_text is not None:
        config_file.write_text(config_text)
    script = """
import json
import logging
import pathlib
import sys

import bentoml
from bentoml._internal.context import server_context

server_context.worker_index = 7
options = {"log_path": sys.argv[2]}
if sys.argv[1] != "-":
    options["log_config_file"] = sys.argv[1]
result = {}
try:
    with bentoml.monitor(sys.argv[3], monitor_options=options) as monitor:
        monitor.log(1, "value", "feature", "numerical")
    handler = logging.getLogger("bentoml_monitor_data").handlers[0]
    result.update(
        status="ok",
        format=handler.formatter._fmt,
        style=handler.formatter._style.default_format[0],
        base_filename=getattr(handler, "baseFilename", None),
    )
except Exception as error:
    result.update(
        status="error",
        error_type=type(error).__name__,
        error=str(error),
    )
finally:
    logging.shutdown()
result["files"] = sorted(
    str(path.relative_to(sys.argv[2]))
    for path in pathlib.Path(sys.argv[2]).rglob("*.log")
)
print(json.dumps(result))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(config_file) if config_text is not None else "-",
            str(tmp_path),
            monitor_name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return t.cast("dict[str, t.Any]", json.loads(result.stdout.splitlines()[-1]))


def logging_config(formatter: dict[str, t.Any]) -> dict[str, t.Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"custom": formatter},
        "handlers": {
            "capture": {
                "class": "logging.FileHandler",
                "filename": "{data_filename}",
                "formatter": "custom",
            }
        },
        "loggers": {
            name: {
                "handlers": ["capture"],
                "level": "INFO",
                "propagate": False,
            }
            for name in LOGGER_NAMES
        },
    }


def test_public_monitor_accepts_literal_json_formatter(tmp_path: Path) -> None:
    result = run_public_monitor(
        tmp_path,
        """
version: 1
disable_existing_loggers: false
formatters:
  custom:
    format: '{"level": "%(levelname)s", "message": "%(message)s"}'
handlers:
  capture:
    class: logging.FileHandler
    filename: '{data_filename}'
    formatter: custom
loggers:
  bentoml_monitor_data:
    handlers: [capture]
    level: INFO
    propagate: false
  bentoml_monitor_schema:
    handlers: [capture]
    level: INFO
    propagate: false
""",
    )

    assert result["status"] == "ok"
    assert result["format"] == ('{"level": "%(levelname)s", "message": "%(message)s"}')
    assert result["files"] == ["brace_probe/data/data.7.log"]


@pytest.mark.parametrize(
    ("config_text", "suffix", "expected_format", "expected_style"),
    [
        (
            """
version: 1
disable_existing_loggers: false
formatters:
  custom: {format: "%(levelname)s %(message)s"}
handlers:
  capture: {class: logging.FileHandler, filename: '{data_filename}', formatter: custom}
loggers:
  bentoml_monitor_data: {handlers: [capture], level: INFO, propagate: false}
  bentoml_monitor_schema: {handlers: [capture], level: INFO, propagate: false}
""",
            "yaml",
            "%(levelname)s %(message)s",
            "%",
        ),
        (
            """
version: 1
disable_existing_loggers: false
formatters:
  custom:
    style: '{'
    format: '{levelname} {message}'
handlers:
  capture: {class: logging.FileHandler, filename: '{data_filename}', formatter: custom}
loggers:
  bentoml_monitor_data: {handlers: [capture], level: INFO, propagate: false}
  bentoml_monitor_schema: {handlers: [capture], level: INFO, propagate: false}
""",
            "yaml",
            "{levelname} {message}",
            "{",
        ),
        (
            json.dumps(
                logging_config(
                    {
                        "format": (
                            '{"level": "%(levelname)s", '
                            '"message": "%(message)s", '
                            '"context": {"logger": "%(name)s"}}'
                        )
                    }
                )
            ),
            "json",
            (
                '{"level": "%(levelname)s", '
                '"message": "%(message)s", '
                '"context": {"logger": "%(name)s"}}'
            ),
            "%",
        ),
    ],
)
def test_public_monitor_accepts_native_logging_braces(
    tmp_path: Path,
    config_text: str,
    suffix: str,
    expected_format: str,
    expected_style: str,
) -> None:
    result = run_public_monitor(tmp_path, config_text, suffix=suffix)

    assert result["status"] == "ok"
    assert result["format"] == expected_format
    assert result["style"] == expected_style


def test_public_monitor_accepts_logging_field_with_known_prefix(
    tmp_path: Path,
) -> None:
    config = logging_config(
        {
            "style": "{",
            "format": "{worker_id_extra} {message}",
            "defaults": {"worker_id_extra": "custom"},
        }
    )

    result = run_public_monitor(tmp_path, json.dumps(config), suffix="json")

    assert result["status"] == "ok"
    assert result["format"] == "{worker_id_extra} {message}"
    assert result["files"] == ["brace_probe/data/data.7.log"]


def test_fallback_replaces_all_bare_tokens_once(tmp_path: Path) -> None:
    config = logging_config(
        {
            "format": (
                '{"schema": "{schema_filename}", '
                '"data": "{data_filename}", '
                '"worker": "{worker_id}", '
                '"monitor": "{monitor_name}", '
                '"message": "%(message)s"}'
            )
        }
    )
    result = run_public_monitor(
        tmp_path,
        json.dumps(config),
        suffix="json",
        monitor_name="outer{worker_id}",
    )
    schema_filename = str(tmp_path / "outer{worker_id}" / "schema" / "schema.7.log")
    data_filename = str(tmp_path / "outer{worker_id}" / "data" / "data.7.log")

    assert result["status"] == "ok"
    assert result["format"] == (
        f'{{"schema": "{schema_filename}", '
        f'"data": "{data_filename}", '
        '"worker": "7", '
        '"monitor": "outer{worker_id}", '
        '"message": "%(message)s"}'
    )
    assert result["base_filename"] == data_filename


def test_fallback_preserves_escaped_brace_semantics(tmp_path: Path) -> None:
    config = logging_config(
        {
            "format": (
                '{{"escaped": "a"}}|{{worker_id}}|'
                '{{{worker_id}}}|{{"worker": {worker_id}}}|'
                '{{{worker_id}: "%(message)s"}}|'
                '{"literal": "json"}|{worker_id}|%(message)s'
            )
        }
    )

    result = run_public_monitor(tmp_path, json.dumps(config), suffix="json")

    assert result["status"] == "ok"
    assert result["format"] == (
        '{"escaped": "a"}|{worker_id}|{7}|{"worker": 7}|'
        '{7: "%(message)s"}|'
        '{"literal": "json"}|7|%(message)s'
    )


def test_legacy_template_behavior_is_unchanged(tmp_path: Path) -> None:
    result = run_public_monitor(
        tmp_path,
        """
version: 1
disable_existing_loggers: false
formatters:
  custom:
    format: >-
      {monitor_name!r}|{monitor_name!s:>20}|{worker_id:03d}|{{monitor_name}}|{{"level"}}|%(message)s
handlers:
  capture:
    class: logging.FileHandler
    filename: '{schema_filename}'
    formatter: custom
loggers:
  bentoml_monitor_data:
    handlers: [capture]
    level: INFO
    propagate: false
  bentoml_monitor_schema:
    handlers: [capture]
    level: INFO
    propagate: false
""",
    )

    assert result["status"] == "ok"
    assert result["format"] == (
        "'brace_probe'|         brace_probe|007|{monitor_name}|{\"level\"}|%(message)s"
    )


@pytest.mark.parametrize(
    "unsupported_filename",
    [
        "{data_filename!r}",
        "{{{data_filename!r}}}",
    ],
)
def test_fallback_rejects_unsupported_known_placeholder_forms(
    tmp_path: Path, unsupported_filename: str
) -> None:
    config = logging_config(
        {"format": '{"level": "%(levelname)s", "message": "%(message)s"}'}
    )
    config["handlers"]["capture"]["filename"] = unsupported_filename

    result = run_public_monitor(tmp_path, json.dumps(config), suffix="json")

    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert "only supports bare BentoML placeholders" in result["error"]
    assert unsupported_filename in result["error"]
    assert result["files"] == []


def test_fallback_rejects_multiline_known_placeholder_modifier(
    tmp_path: Path,
) -> None:
    result = run_public_monitor(
        tmp_path,
        """
version: 1
disable_existing_loggers: false
formatters:
  custom:
    format: '{"level": "%(levelname)s", "message": "%(message)s"}'
handlers:
  capture:
    class: logging.FileHandler
    filename: >-
      {data_filename:
      >20}
    formatter: custom
loggers:
  bentoml_monitor_data: {handlers: [capture], level: INFO, propagate: false}
  bentoml_monitor_schema: {handlers: [capture], level: INFO, propagate: false}
""",
    )

    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert "only supports bare BentoML placeholders" in result["error"]
    assert result["files"] == []


def test_default_config_still_exports_monitor_data(tmp_path: Path) -> None:
    result = run_public_monitor(tmp_path, None)

    assert result["status"] == "ok"
    assert result["files"] == [
        "brace_probe/data/data.7.log",
        "brace_probe/schema/schema.7.log",
    ]
