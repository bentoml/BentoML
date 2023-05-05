from __future__ import annotations

import typing as t
import logging
import logging.config
from functools import lru_cache

from .context import trace_context
from .context import component_context
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
            "propagate": False,
        },
    },
    "root": {"level": logging.WARNING},
}

TRACED_LOG_FORMAT = (
    "%(asctime)s %(levelname_bracketed)s %(component)s %(message)s%(trace_msg)s"
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
            "stream": "ext://sys.stdout",
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


@lru_cache(maxsize=1)
def _component_name():
    result = ""
    if component_context.component_type:
        result = component_context.component_type
    if component_context.component_name:
        result = f"{result}:{component_context.component_name}"
    if component_context.component_index:
        result = f"{result}:{component_context.component_index}"
    return result


def trace_record_factory(*args: t.Any, **kwargs: t.Any):
    record = default_factory(*args, **kwargs)
    if record.name in ("bentoml_monitor_data", "bentoml_monitor_schema"):
        return record
    record.levelname_bracketed = f"[{record.levelname}]"
    record.component = f"[{_component_name()}]"
    trace_id = trace_context.trace_id
    if trace_id in (0, None):
        record.trace_msg = ""
    else:
        from .configuration.containers import BentoMLContainer

        logging_formatting = BentoMLContainer.logging_formatting.get()
        trace_id_format = logging_formatting["trace_id"]
        span_id_format = logging_formatting["span_id"]

        trace_id = format(trace_id, trace_id_format)
        span_id = format(trace_context.span_id, span_id_format)
        record.trace_msg = f" (trace={trace_id},span={span_id},sampled={trace_context.sampled},service.name={trace_context.service_name})"
    record.request_id = trace_context.request_id
    record.service_name = trace_context.service_name

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
