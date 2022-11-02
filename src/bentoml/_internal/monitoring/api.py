from __future__ import annotations

import typing as t
import logging
import datetime
import contextlib
import collections
import logging.config
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
from simple_di import inject
from simple_di import Provide

from ..types import LazyType
from ..context import component_context
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..types import JSONSerializable

DT = t.TypeVar("DT")
MT = t.TypeVar("MT", bound="MonitorBase[t.Any]")

logger = logging.getLogger(__name__)

BENTOML_MONITOR_ROLES = {"feature", "prediction", "target"}
BENTOML_MONITOR_TYPES = {"numerical", "categorical", "numerical_sequence"}

MONITOR_REGISTRY: dict[str, MonitorBase[t.Any]] = {}  # cache of monitors


class MonitorBase(t.Generic[DT]):
    def __init__(self, name: str, **kwargs: t.Any) -> None:
        raise NotImplementedError()

    def start_record(self) -> None:
        raise NotImplementedError()

    def stop_record(self) -> None:
        raise NotImplementedError()

    def export_schema(self) -> JSONSerializable:
        raise NotImplementedError()

    def export_data(self) -> JSONSerializable:
        raise NotImplementedError()

    def log(self, data: DT, name: str, role: str, data_type: str) -> None:
        raise NotImplementedError()

    def log_batch(
        self,
        data_batch: t.Iterable[DT],
        name: str,
        role: str,
        data_type: str,
    ) -> None:
        raise NotImplementedError()

    def log_table(
        self,
        data: t.Iterable[t.Iterable[DT]],
        schema: dict[str, str],
    ) -> None:
        raise NotImplementedError()


class NoOpMonitor(MonitorBase[t.Any]):
    def __init__(self, name: str, **kwargs: t.Any) -> None:
        pass

    def start_record(self) -> None:
        pass

    def stop_record(self) -> None:
        pass

    def export_schema(self) -> JSONSerializable:
        pass

    def export_data(self) -> JSONSerializable:
        pass

    def log(self, data: t.Any, name: str, role: str, data_type: str) -> None:
        pass

    def log_batch(
        self,
        data_batch: t.Iterable[t.Any],
        name: str,
        role: str,
        data_type: str,
    ) -> None:
        pass

    def log_table(
        self,
        data: t.Iterable[t.Iterable[t.Any]],
        schema: dict[str, str],
    ) -> None:
        pass


DEFAULT_CONFIG_YAML = """
version: 1
incremental: false
loggers:
  bentoml_monitor_data:
    level: INFO
    handlers: [bentoml_monitor_data]
  bentoml_monitor_schema:
    level: INFO
    handlers: [bentoml_monitor_schema]
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
    format: " "
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
    ) -> None:
        self.name = name
        self.log_config_file = log_config_file
        self.log_path = log_path
        self._is_first_record = True
        self._is_first_column = False
        self._column_meta: dict[str, dict[str, str]] = {}
        self._columns: dict[
            str,
            collections.deque[JSONSerializable],
        ] = collections.defaultdict(collections.deque)

    def _init_logger(self) -> None:
        if self.log_config_file is None:
            logging_config_yaml = DEFAULT_CONFIG_YAML
        else:
            with open(self.log_config_file, "r", encoding="utf8") as f:
                logging_config_yaml = f.read()

        worker_id = component_context.component_index
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
        # pylint: disable=attribute-defined-outside-init
        self.data_logger = logging.getLogger("bentoml_monitor_data")
        self.schema_logger = logging.getLogger("bentoml_monitor_schema")
        self.data_logger.propagate = False
        self.schema_logger.propagate = False

    def start_record(self) -> None:
        """
        Start recording data. This method should be called before logging any data.
        """
        self._is_first_column = True

    def stop_record(self) -> None:
        """
        Stop recording data. This method should be called after logging all data.
        """
        if self._is_first_record:
            self._init_logger()
            self.export_schema()
            self._is_first_record = False

        if self._is_first_column:
            logger.warning("No data logged in this record. Will skip this record.")
        else:
            self.export_data()

    def export_schema(self):
        """
        Export schema of the data. This method should be called after all data is logged.
        """
        from bentoml._internal.context import component_context

        self.schema_logger.info(
            dict(
                meta_data={
                    "bento_name": component_context.bento_name,
                    "bento_version": component_context.bento_version,
                },
                columns=list(self._column_meta.values()),
            )
        )

    def export_data(self):
        """
        Export data. This method should be called after all data is logged.
        """
        assert (
            len(set(len(q) for q in self._columns.values())) == 1
        ), "All columns must have the same length"
        while True:
            try:
                record = {k: v.popleft() for k, v in self._columns.items()}
                self.data_logger.info(record)
            except IndexError:
                break

    def log(
        self,
        data: JSONSerializable,
        name: str,
        role: str,
        data_type: str,
        ignore_existing: bool = False,
    ) -> None:
        """
        log a data with column name, role and type to the current record
        """
        if name in self.PRESERVED_COLUMNS:
            raise ValueError(
                f"Column name {name} is preserved. Please use a different name."
            )

        assert role in BENTOML_MONITOR_ROLES, f"Invalid role {role}"
        assert data_type in BENTOML_MONITOR_TYPES, f"Invalid data type {data_type}"

        if self._is_first_record:
            if name not in self._column_meta:
                self._column_meta[name] = {
                    "name": name,
                    "role": role,
                    "type": data_type,
                }
            elif not ignore_existing:
                raise ValueError(
                    f"Column {name} already exists. Please use a different name."
                )
        if self._is_first_column:
            self._is_first_column = False

            from ..context import trace_context

            # universal columns
            self._columns[self.COLUMN_TIME].append(datetime.datetime.now().isoformat())
            self._columns[self.COLUMN_RID].append(trace_context.request_id)

        self._columns[name].append(data)

    def log_batch(
        self,
        data_batch: t.Iterable[JSONSerializable],
        name: str,
        role: str,
        data_type: str,
    ) -> None:
        """
        Log a batch of data. The data will be logged as a single column.
        """
        try:
            for data in data_batch:
                self.log(data, name, role, data_type, ignore_existing=True)
        except TypeError:
            raise ValueError(
                "data_batch is not iterable. Please use log() to log a single data."
            ) from None

    def log_table(
        self,
        data: t.Iterable[t.Iterable[JSONSerializable]],
        schema: dict[str, str],
    ) -> None:
        raise NotImplementedError("Not implemented yet")


@t.overload
def monitor(
    name: str,
    monitor_class: t.Type[MT],
    monitor_options: dict[str, t.Any] | None,
) -> t.Generator[MT, None, None]:
    pass


@t.overload
def monitor(
    name: str,
    monitor_class: str,
    monitor_options: dict[str, t.Any] | None,
) -> t.Generator[MonitorBase[t.Any], None, None]:
    pass


@t.overload
def monitor(
    name: str,
    monitor_class: None,
    monitor_options: dict[str, t.Any] | None,
) -> t.Generator[MonitorBase[t.Any], None, None]:
    pass


@contextlib.contextmanager
@inject
def monitor(
    name: str,
    monitor_class: t.Type[MT]
    | str
    | None = Provide[BentoMLContainer.config.monitoring.type],
    monitor_options: dict[str, t.Any]
    | None = Provide[BentoMLContainer.config.monitoring.options],
) -> t.Generator[MT | MonitorBase[t.Any], None, None]:
    """
    Context manager for monitoring.

    :param name: name of the monitor
    :param monitor_class: class of the monitor

    :return: a monitor instance

    Example::

        with bentoml.monitor("my_monitor") as m:
            m.log(1, "x", "feature", "numerical")
            m.log(2, "y", "feature", "numerical")
            m.log(3, "z", "feature", "numerical")
            m.log(4, "prediction", "prediction", "numerical")

        # or
        with bentoml.monitor("my_monitor") as m:
            m.log_batch([1, 2, 3], "x", "feature", "numerical")
            m.log_batch([4, 5, 6], "y", "feature", "numerical")
            m.log_batch([7, 8, 9], "z", "feature", "numerical")
            m.log_batch([10, 11, 12], "prediction", "prediction", "numerical")

        this will log the following data:
        {
            "timestamp": "2021-09-01T00:00:00",
            "request_id": "1234567890",
            "x": 1,
            "y": 2,
            "z": 3,
            "prediction": 4,
        }
        and the following schema:
        [
            {"name": "timestamp", "role": "time", "type": "datetime"},
            {"name": "request_id", "role": "request_id", "type": "string"},
            {"name": "x", "role": "feature", "type": "numerical"},
            {"name": "y", "role": "feature", "type": "numerical"},
            {"name": "z", "role": "feature", "type": "numerical"},
            {"name": "prediction", "role": "prediction", "type": "numerical"},
        ]
    """
    if name not in MONITOR_REGISTRY:
        if not BentoMLContainer.config.monitoring.enabled.get():
            monitor_klass = NoOpMonitor
        elif monitor_class is None or monitor_class == "default":
            monitor_klass = DefaultMonitor
        elif isinstance(monitor_class, str):
            monitor_klass = LazyType["MonitorBase[t.Any]"](monitor_class).get_class()
        elif isinstance(monitor_class, type):
            monitor_klass = monitor_class
        else:
            raise ValueError(
                "monitor_class must be a class, a string or None. "
                f"Got {type(monitor_class)}"
            )
        if monitor_options is None:
            monitor_options = {}
        MONITOR_REGISTRY[name] = monitor_klass(name, **monitor_options)

    mon = MONITOR_REGISTRY[name]
    mon.start_record()
    yield mon
    mon.stop_record()
