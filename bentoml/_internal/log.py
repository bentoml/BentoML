import typing
import logging
import logging.config
from logging import Filter
from logging import Formatter

from .trace import ServiceContext
from .configuration import get_debug_mode


class TraceFilter(Filter):
    """
    Logging filter implementation that injects tracing related fields to the log record.
    """

    def filter(self, record):
        record.sampled = ServiceContext.sampled
        record.trace_id = ServiceContext.trace_id
        record.span_id = ServiceContext.span_id
        record.request_id = ServiceContext.request_id
        record.component = ServiceContext.component_name

        return Filter.filter(self, record)


TRACED_LOG_FORMAT = "[%(component)s] %(message)s (trace=%(trace_id)s,span=%(span_id)s,sampled=%(sampled)s)"
SERVICE_LOG_FORMAT = "[%(component)s] %(message)s"
DATE_FORMAT = "%x %X"


class TraceFormatter(Formatter):
    """
    Logging formatter implementation that picks format dynamically based ont he presence
    of the tracing related fields. If present, the formatter will include trace and span
    IDs in the log message.
    """

    def __init__(self):
        Formatter.__init__(
            self,
            fmt=TRACED_LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
        self.control_formmater = Formatter(SERVICE_LOG_FORMAT, datefmt=DATE_FORMAT)
        self.trace_formatter = Formatter(TRACED_LOG_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record):
        if record.trace_id == 0:
            return self.control_formmater.format(record)
        else:
            return self.trace_formatter.format(record)


LOGGING_CONFIG: typing.Dict[str, typing.Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"tracing": {"()": "bentoml._internal.log.TraceFormatter"}},
    "filters": {
        "tracing": {
            "()": "bentoml._internal.log.TraceFilter",
        },
    },
    "handlers": {
        "internal": {
            "level": "INFO",
            "filters": ["tracing"],
            "formatter": "tracing",
            "()": "rich.logging.RichHandler",
            "omit_repeated_times": False,
            "rich_tracebacks": True,
            "show_path": get_debug_mode(),  # show log line # in debug mode
        },
        "uvicorn": {
            "level": "INFO",
            "filters": ["tracing"],
            "formatter": "tracing",
            "()": "rich.logging.RichHandler",
            "omit_repeated_times": False,
            "rich_tracebacks": True,
            "show_path": get_debug_mode(),  # show log line # in debug mode
        },
    },
    "loggers": {
        "bentoml": {
            "handlers": ["internal"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn": {"handlers": [], "level": "INFO"},
        "uvicorn.error": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": [], "level": "INFO", "propagate": False},
    },
}


def configure_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
