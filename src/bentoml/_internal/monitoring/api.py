from __future__ import annotations

import typing as t
import logging
import contextlib
import collections
import contextvars
import logging.config
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from ..types import LazyType
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from .default import DefaultMonitor

DT = t.TypeVar("DT")
MT = t.TypeVar("MT", bound="MonitorBase[t.Any]")

logger = logging.getLogger(__name__)

BENTOML_MONITOR_ROLES = {"feature", "prediction", "target"}
BENTOML_MONITOR_TYPES = {"numerical", "categorical", "numerical_sequence"}

MONITOR_REGISTRY: dict[str, MonitorBase[t.Any]] = {}  # cache of monitors

MON_COLUMN_VAR: contextvars.ContextVar[
    "dict[str, dict[str, str]] | None"
] = contextvars.ContextVar("MON_COLUMN_VAR", default=None)
MON_DATAS_VAR: contextvars.ContextVar[
    "dict[str, collections.deque[t.Any]] | None"
] = contextvars.ContextVar("MON_DATAS_VAR", default=None)


class MonitorBase(t.Generic[DT]):
    """
    The base monitor class. All monitors should inherit from this class.
    Subclasses should implement the following methods:
    - export_schema
    - export_data
    to export the schema and data to the desired format.
    """

    PRESERVED_COLUMNS: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        **_: t.Any,
    ) -> None:
        self.name = name
        self.columns_schema: dict[str, dict[str, str]] | None = None

    def start_record(self):
        """
        Start recording data. This method should be called before logging any data.
        """
        if self.columns_schema is None:
            assert MON_COLUMN_VAR.get() is None
            MON_COLUMN_VAR.set({})

        assert MON_DATAS_VAR.get() is None
        MON_DATAS_VAR.set(collections.defaultdict(collections.deque))

    def stop_record(self) -> None:
        """
        Stop recording data. This method should be called after logging all data.
        """
        datas: dict[str, collections.deque[DT]] = MON_DATAS_VAR.get()  # type: ignore
        assert datas is not None

        if len(datas) == 0:
            logger.warning("No data logged in this record. Will skip output.")
            return

        if self.columns_schema is None:
            columns = MON_COLUMN_VAR.get()
            assert columns is not None
            self.columns_schema = columns
            self.export_schema(columns)
            MON_COLUMN_VAR.set(None)

        if len(set(len(q) for q in datas.values())) != 1:
            assert ValueError("All columns should have the same length.")
        self.export_data(datas)

        MON_DATAS_VAR.set(None)

    def export_schema(self, columns_schema: dict[str, dict[str, str]]) -> None:
        """
        Export schema of the data. This method should be called after all data is logged.
        """
        raise NotImplementedError()

    def export_data(self, datas: dict[str, collections.deque[DT]]) -> None:
        """
        Export data. This method should be called after all data is logged.
        """
        raise NotImplementedError()

    def log(
        self,
        data: DT,
        name: str,
        role: str,
        data_type: str,
    ) -> None:
        """
        log a data with column name, role and type to the current record
        """
        if name in self.PRESERVED_COLUMNS:
            logger.warning(
                "Column name %s is reserved, will be renamed to %s", name, name + "_"
            )
            name = name + "_"

        if role not in BENTOML_MONITOR_ROLES:
            logger.warning(
                "Role {role} is not officially supported, but will be logged anyway."
            )
        if data_type not in BENTOML_MONITOR_TYPES:
            logger.warning(
                "Data type {data_type} is not officially supported, but will be logged anyway."
            )

        if self.columns_schema is None:
            columns = MON_COLUMN_VAR.get()
            assert columns is not None
            if name in columns:
                logger.warning(
                    "Column name %s is duplicated, will be ignored.",
                    name,
                )
            else:
                columns[name] = {
                    "name": name,
                    "role": role,
                    "type": data_type,
                }
        datas = MON_DATAS_VAR.get()
        assert datas is not None
        datas[name].append(data)

    def log_batch(
        self,
        data_batch: t.Iterable[DT],
        name: str,
        role: str,
        data_type: str,
    ) -> None:
        """
        Log a batch of data. The data will be logged as a single column.
        """
        try:
            for data in data_batch:
                self.log(data, name, role, data_type)
        except TypeError:
            logger.warning(
                "Data batch is not iterable, will ignore the data batch. Please use log() to log a single data."
            )

    def log_table(
        self,
        data: t.Iterable[t.Iterable[DT]],
        schema: dict[str, str],
    ) -> None:
        logger.warning(
            "log_table() is not implemented yet. Will ignore the data. Please use log() or log_batch() instead."
        )
        return


class NoOpMonitor(MonitorBase[t.Any]):
    def __init__(self, name: str, **kwargs: t.Any) -> None:
        pass

    def start_record(self) -> None:
        pass

    def stop_record(self) -> None:
        pass

    def export_schema(self, columns_schema: dict[str, dict[str, str]]) -> None:
        pass

    def export_data(self, datas: dict[str, collections.deque[t.Any]]) -> None:
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


@t.overload
@contextlib.contextmanager
def monitor(
    name: str | t.Any,
    monitor_class: DefaultMonitor = ...,
    monitor_options: dict[str, t.Any] | None = ...,
) -> t.Generator[DefaultMonitor, None, None]:
    ...


@t.overload
@contextlib.contextmanager
def monitor(
    name: str | t.Any,
    monitor_class: str = ...,
    monitor_options: dict[str, t.Any] | None = ...,
) -> t.Generator[MonitorBase[t.Any], None, None]:
    ...


@t.overload
@contextlib.contextmanager
def monitor(
    name: str | t.Any,
    monitor_class: None = ...,
    monitor_options: dict[str, t.Any] | None = ...,
) -> t.Generator[MonitorBase[t.Any], None, None]:
    ...


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
    :param monitor_class: class of the monitor, can be a string or a class
        example:
        - default
        - otlp
        - "bentoml.monitoring.prometheus.PrometheusMonitor"
    :param monitor_options: options for the monitor

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
            from .default import DefaultMonitor

            monitor_klass = DefaultMonitor
        elif monitor_class == "otlp":
            from .otlp import OTLPMonitor

            monitor_klass = OTLPMonitor
        elif isinstance(monitor_class, str):
            monitor_klass = LazyType["MonitorBase[t.Any]"](monitor_class).get_class()
        elif isinstance(monitor_class, type):
            monitor_klass = monitor_class
        else:
            logger.warning(
                "Invalid monitor class, will disable monitoring. Please check your configuration."
            )
            monitor_klass = NoOpMonitor
        if monitor_options is None:
            monitor_options = {}
        MONITOR_REGISTRY[name] = monitor_klass(name, **monitor_options)

    mon = MONITOR_REGISTRY[name]
    mon.start_record()
    yield mon
    mon.stop_record()
