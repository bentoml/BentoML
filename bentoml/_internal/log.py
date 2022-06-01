from __future__ import annotations

import typing as t
import logging
import logging.config
from logging import Filter
from logging import Formatter

import psutil

from .trace import ServiceContext
from .configuration import get_debug_mode

TRACED_LOG_FORMAT = "[%(component)s] %(message)s (trace=%(trace_id)s,span=%(span_id)s,sampled=%(sampled)s)"
SERVICE_LOG_FORMAT = "[%(component)s] %(message)s"
DATE_FORMAT = "%x %X"


class TraceFilter(Filter):
    """
    Logging filter implementation that injects tracing related fields to the log record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if "circus" in record.name:
            component_name = "circus"
        elif "asyncio" in record.name:
            component_name = "asyncio"
        else:
            component_name = ServiceContext.component_name

        # make it type-safe
        object.__setattr__(record, "sampled", ServiceContext.sampled)
        object.__setattr__(record, "trace_id", ServiceContext.trace_id)
        object.__setattr__(record, "span_id", ServiceContext.span_id)
        object.__setattr__(record, "request_id", ServiceContext.request_id)
        object.__setattr__(record, "component", component_name)

        return Filter.filter(self, record)


class TraceFormatter(Formatter):
    """
    Logging formatter implementation that picks format dynamically based ont he presence
    of the tracing related fields. If present, the formatter will include trace and span
    IDs in the log message.
    """

    def __init__(self):
        Formatter.__init__(self, fmt=TRACED_LOG_FORMAT, datefmt=DATE_FORMAT)
        self.control_formatter = Formatter(SERVICE_LOG_FORMAT, datefmt=DATE_FORMAT)
        self.trace_formatter = Formatter(TRACED_LOG_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if record.trace_id == 0:
            return self.control_formatter.format(record)
        else:
            return self.trace_formatter.format(record)


if psutil.WINDOWS:
    # required by rich logging Handler
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

LOGGING_CONFIG: dict[str, t.Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"tracing": {"()": "bentoml._internal.log.TraceFormatter"}},
    "filters": {
        "tracing": {"()": "bentoml._internal.log.TraceFilter"},
    },
    "handlers": {
        "internal": {
            "level": "INFO",
            "filters": ["tracing"],
            "formatter": "tracing",
            "()": "rich.logging.RichHandler",
            "omit_repeated_times": False,
            "rich_tracebacks": True,
            "tracebacks_show_locals": get_debug_mode(),
            "show_path": get_debug_mode(),  # show log line # in debug mode
        },
        "uvicorn": {
            "level": "INFO",
            "filters": ["tracing"],
            "formatter": "tracing",
            "()": "rich.logging.RichHandler",
            "omit_repeated_times": False,
            "rich_tracebacks": True,
            "tracebacks_show_locals": get_debug_mode(),
            "show_path": get_debug_mode(),  # show log line # in debug mode
        },
        "circus": {
            "level": "INFO",
            "filters": ["tracing"],
            "formatter": "tracing",
            "()": "rich.logging.RichHandler",
            "omit_repeated_times": True,
            "rich_tracebacks": True,
            "tracebacks_show_locals": get_debug_mode(),
            "show_path": get_debug_mode(),  # show log line # in debug mode
        },
    },
    "loggers": {
        "bentoml": {"handlers": ["internal"], "level": "INFO", "propagate": False},
        # circus logger
        "circus": {"handlers": ["circus"], "level": "INFO", "propagate": False},
        "circus.plugins": {"handlers": ["circus"], "level": "INFO", "propagate": False},
        "asyncio": {"handlers": ["internal"], "level": "INFO", "propagate": False},
        # uvicorn logger
        "uvicorn": {"handlers": [], "level": "INFO"},
        "uvicorn.error": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": [], "level": "INFO", "propagate": False},
    },
}


def configure_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
