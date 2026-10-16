from __future__ import annotations

import collections
import datetime
import logging
import logging.config
import re
import typing as t
from pathlib import Path

import yaml
from yaml.tokens import ScalarToken

from ..context import server_context
from ..context import trace_context
from .base import MonitorBase

if t.TYPE_CHECKING:
    from ..types import JSONSerializable


DEFAULT_CONFIG_YAML = """
version: 1
disable_existing_loggers: false
loggers:
  bentoml_monitor_data:
    level: INFO
    handlers: [bentoml_monitor_data]
    propagate: false
  bentoml_monitor_schema:
    level: INFO
    handlers: [bentoml_monitor_schema]
    propagate: false
handlers:
  bentoml_monitor_data:
    class: logging.handlers.TimedRotatingFileHandler
    level: INFO
    formatter: bentoml_json
    filename: '{data_filename}'
    when: "D"
  bentoml_monitor_schema:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: bentoml_json
    filename: '{schema_filename}'
formatters:
  bentoml_json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "()"
    validate: false
"""

_FALLBACK_TOKEN = re.compile(
    r"(?P<wrapped>{{{(?P<wrapped_name>"
    r"schema_filename|data_filename|worker_id|monitor_name)}}})|"
    r"(?P<escaped>{{|}})|"
    r"{(?P<name>schema_filename|data_filename|worker_id|monitor_name)}"
)
_KNOWN_BRACED_VALUE = re.compile(
    r"(?P<open>{+)(?P<name>schema_filename|data_filename|worker_id|monitor_name)"
    r"(?=}|!|:|\.|\[)"
    r"(?P<suffix>[^}]*)}(?P<close>}*)"
)


def _format_fallback_tokens(value: str, parameters: dict[str, t.Any]) -> str:
    for match in _KNOWN_BRACED_VALUE.finditer(value):
        if len(match.group("open")) % 2 and match.group("suffix"):
            raise ValueError(
                "Logging config fallback only supports bare BentoML placeholders; "
                f"unsupported placeholder: {match.group(0)!r}"
            )

    try:
        scalar_ranges = [
            (token.start_mark.index, token.end_mark.index)
            for token in yaml.scan(value)
            if isinstance(token, ScalarToken)
        ]
    except yaml.YAMLError:
        scalar_ranges = []

    escaped_opens: dict[tuple[int, int], int] = {}

    def replace(match: re.Match[str]) -> str:
        if match.group("wrapped") is not None:
            return f"{{{parameters[match.group('wrapped_name')]}}}"
        if match.group("escaped") is not None:
            scalar_range = next(
                (
                    (start, end)
                    for start, end in scalar_ranges
                    if start <= match.start() < end
                ),
                None,
            )
            # Adjacent braces outside scalar values can be JSON/YAML structure.
            if scalar_range is None:
                return match.group(0)
            if match.group(0) == "{{":
                escaped_opens[scalar_range] = escaped_opens.get(scalar_range, 0) + 1
                return match.group(0)[0]
            if escaped_opens.get(scalar_range, 0):
                escaped_opens[scalar_range] -= 1
                return match.group(0)[0]
            return match.group(0)
        return str(parameters[match.group("name")])

    return _FALLBACK_TOKEN.sub(replace, value)


class DefaultMonitor(MonitorBase["JSONSerializable"]):
    """
    The default monitor implementation. It uses a logger to log data and schema, and will
    write monitor data to rotating files. The schema is logged as a JSON object, and the
    data is logged as a JSON array.
    """

    PRESERVED_COLUMNS = (COLUMN_TIME, COLUMN_RID, COLUMN_TID) = (
        "timestamp",
        "request_id",
        "trace_id",
    )

    def __init__(
        self,
        name: str,
        log_path: str,
        log_config_file: str | None = None,
        **_: t.Any,
    ) -> None:
        super().__init__(name, **_)
        self.log_config_file = log_config_file
        self.log_path = log_path
        self.data_logger = None
        self.schema_logger = None

    def _init_logger(self) -> None:
        if self.log_config_file is None:
            logging_config_yaml = DEFAULT_CONFIG_YAML
        else:
            with open(self.log_config_file, "r", encoding="utf8") as f:
                logging_config_yaml = f.read()

        worker_id = server_context.worker_index or 0
        schema_path = Path(self.log_path).joinpath(
            self.name, "schema", f"schema.{worker_id}.log"
        )
        data_path = Path(self.log_path).joinpath(
            self.name, "data", f"data.{worker_id}.log"
        )

        schema_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        parameters = dict(
            schema_filename=str(schema_path.absolute()),
            data_filename=str(data_path.absolute()),
            worker_id=worker_id,
            monitor_name=self.name,
        )

        try:
            logging_config_yaml = logging_config_yaml.format(**parameters)
        except (AttributeError, IndexError, KeyError, ValueError) as format_error:
            try:
                logging_config_yaml = _format_fallback_tokens(
                    logging_config_yaml, parameters
                )
            except ValueError as e:
                raise e from format_error

        try:
            logging_config = yaml.safe_load(logging_config_yaml)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Error loading logging config from {self.log_config_file}: {e}"
            ) from e

        logging.config.dictConfig(logging_config)
        self.data_logger = logging.getLogger("bentoml_monitor_data")
        self.schema_logger = logging.getLogger("bentoml_monitor_schema")

    def export_schema(self, columns_schema: dict[str, dict[str, str]]) -> None:
        """
        Export columns_schema of the data. This method should be called after all data is logged.
        """
        if self.schema_logger is None:
            self._init_logger()
            assert self.schema_logger is not None

        self.schema_logger.info(
            dict(
                meta_data={
                    "bento_name": server_context.bento_name,
                    "bento_version": server_context.bento_version,
                },
                columns=list(columns_schema.values()),
            )
        )

    def export_data(
        self,
        datas: dict[str, collections.deque[JSONSerializable]],
    ) -> None:
        """
        Export data. This method should be called after all data is logged.
        """
        if self.data_logger is None:
            self._init_logger()
            assert self.data_logger is not None

        extra_columns = {
            self.COLUMN_TIME: datetime.datetime.now().isoformat(),
            self.COLUMN_RID: str(trace_context.request_id),
            self.COLUMN_TID: str(trace_context.trace_id),
        }
        while True:
            try:
                record = {k: v.popleft() for k, v in datas.items()}
                record.update(extra_columns)
                self.data_logger.info(record)
            except IndexError:
                break
