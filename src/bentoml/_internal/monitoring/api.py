from __future__ import annotations

import typing as t
import logging
import contextlib
import logging.config

from simple_di import inject
from simple_di import Provide

from .base import MT
from .base import MonitorBase
from .base import NoOpMonitor
from ..types import LazyType
from .default import DefaultMonitor
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

_MONITOR_INSTANCES: dict[str, MonitorBase[t.Any]] = {}  # cache of monitors


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
    monitor_class: type[MT]
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
    if name not in _MONITOR_INSTANCES:
        if not BentoMLContainer.config.monitoring.enabled.get():
            monitor_klass = NoOpMonitor
        elif monitor_class is None or monitor_class == "default":
            logger.debug("No monitor class is provided, will use default monitor.")
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
                "Invalid monitor class, will disable monitoring. Please check your configuration. Setting monitor class to NoOp."
            )
            monitor_klass = NoOpMonitor

        if monitor_options is None:
            monitor_options = {}

        _MONITOR_INSTANCES[name] = monitor_klass(name, **monitor_options)

    mon = _MONITOR_INSTANCES[name]
    mon.start_record()
    yield mon
    mon.stop_record()
