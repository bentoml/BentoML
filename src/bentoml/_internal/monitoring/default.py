from __future__ import annotations

import typing as t
import logging
import datetime
import collections
import logging.config
from typing import TYPE_CHECKING
from pathlib import Path

import yaml

from .api import MonitorBase
from ..context import trace_context
from ..context import component_context

if TYPE_CHECKING:
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
    filename: "{data_filename}"
    when: "D"
  bentoml_monitor_schema:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: bentoml_json
    filename: "{schema_filename}"
formatters:
  bentoml_json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "()"
    validate: false
"""


class DefaultMonitor(MonitorBase["JSONSerializable"]):
    """
    The default monitor implementation. It uses a logger to log data and schema, and will
    write monitor data to rotating files. The schema is logged as a JSON object, and the
    data is logged as a JSON array.
    """

    PRESERVED_COLUMNS = (COLUMN_TIME, COLUMN_RID) = ("timestamp", "request_id")

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

        worker_id = component_context.component_index or 0
        schema_path = Path(f"{self.log_path}/{self.name}/schema/schema.{worker_id}.log")
        data_path = Path(f"{self.log_path}/{self.name}/data/data.{worker_id}.log")

        schema_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        logging_config_yaml = logging_config_yaml.format(
            schema_filename=str(schema_path.absolute()),
            data_filename=str(data_path.absolute()),
            worker_id=worker_id,
            monitor_name=self.name,
        )

        logging_config = yaml.safe_load(logging_config_yaml)
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
                    "bento_name": component_context.bento_name,
                    "bento_version": component_context.bento_version,
                },
                columns=list(columns_schema.values()),
            )
        )

    def export_data(
        self,
        datas: t.Dict[str, collections.deque[JSONSerializable]],
    ) -> None:
        """
        Export data. This method should be called after all data is logged.
        """
        if self.data_logger is None:
            self._init_logger()
            assert self.data_logger is not None

        extra_columns = dict(
            timestamp=datetime.datetime.now().isoformat(),
            request_id=str(trace_context.request_id),
        )
        while True:
            try:
                record = {k: v.popleft() for k, v in datas.items()}
                record.update(extra_columns)
                self.data_logger.info(record)
            except IndexError:
                break
