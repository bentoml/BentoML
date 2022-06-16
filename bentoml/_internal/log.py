from __future__ import annotations

import typing as t
import logging
import logging.config

from .trace import ServiceContext
from .configuration import get_debug_mode
from .configuration import get_quiet_mode

default_factory = logging.getLogRecordFactory()


# TODO: remove this filter after implementing CLI output as something other than INFO logs
class InfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return logging.INFO <= record.levelno < logging.WARNING


# TODO: can be removed after the above is complete
CLI_LOGGING_CONFIG: dict[str, t.Any] = {
    "version": 1,
    "disable_existing_loggers": True,
    "filters": {"infofilter": {"()": InfoFilter}},
    "handlers": {
        "bentomlhandler": {
            "class": "logging.StreamHandler",
            "filters": ["infofilter"],
            "stream": "ext://sys.stdout",
        },
        "defaulthandler": {
            "class": "logging.StreamHandler",
            "level": logging.WARNING,
        },
    },
    "loggers": {
        "bentoml": {
            "handlers": ["bentomlhandler", "defaulthandler"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
    "root": {"level": logging.WARNING},
}

TRACED_LOG_FORMAT = (
    "%(asctime)s %(levelname_bracketed)s %(name)s %(component)s %(message)s%(trace_msg)s"
)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

SERVER_LOGGING_CONFIG: dict[str, t.Any] = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "traced": {
            "format": TRACED_LOG_FORMAT,
            "datefmt": DATE_FORMAT,
        }
    },
    "handlers": {
        "tracehandler": {
            "class": "logging.StreamHandler",
            "formatter": "traced",
        },
    },
    "loggers": {
        "bentoml": {
            "level": logging.INFO,
            "handlers": ["tracehandler"],
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["tracehandler"],
        "level": logging.WARNING,
    },
}


def configure_logging():
    # TODO: convert to simple 'logging.basicConfig' after we no longer need the filter
    if get_quiet_mode():
        CLI_LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.ERROR
        CLI_LOGGING_CONFIG["root"]["level"] = logging.ERROR
    elif get_debug_mode():
        CLI_LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.DEBUG
        CLI_LOGGING_CONFIG["root"]["level"] = logging.DEBUG
    else:
        CLI_LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.INFO
        CLI_LOGGING_CONFIG["root"]["level"] = logging.WARNING

    logging.config.dictConfig(CLI_LOGGING_CONFIG)


def trace_record_factory(*args: t.Any, **kwargs: t.Any):
    record = default_factory(*args, **kwargs)
    record.levelname_bracketed = f"[{record.levelname}]"  # type: ignore (adding fields to record)
    record.component = f"[{ServiceContext.component_name}]"  # type: ignore (adding fields to record)
    if ServiceContext.trace_id == 0:
        record.trace_msg = ""  # type: ignore (adding fields to record)
    else:
        record.trace_msg = f" (trace={ServiceContext.trace_id},span={ServiceContext.span_id},sampled={ServiceContext.sampled})"  # type: ignore (adding fields to record)
    record.request_id = ServiceContext.request_id  # type: ignore (adding fields to record)

    return record


def configure_server_logging():
    if get_quiet_mode():
        SERVER_LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.ERROR
        SERVER_LOGGING_CONFIG["root"]["level"] = logging.ERROR
    elif get_debug_mode():
        SERVER_LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.DEBUG
        SERVER_LOGGING_CONFIG["root"]["level"] = logging.DEBUG
    else:
        SERVER_LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.INFO
        SERVER_LOGGING_CONFIG["root"]["level"] = logging.WARNING

    logging.setLogRecordFactory(trace_record_factory)
    logging.config.dictConfig(SERVER_LOGGING_CONFIG)
