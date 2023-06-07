from __future__ import annotations

import typing as t
import logging
import importlib

from .api import monitor
from .base import MonitorBase
from .base import NoOpMonitor
from .default import DefaultMonitor
from ...exceptions import MissingDependencyException

logger = logging.getLogger(__name__)

_is_otlp_available = False
try:
    from .otlp import OTLPMonitor as OTLPMonitor

    _is_otlp_available = True
except (ImportError, MissingDependencyException):
    # NOTE: we want this here so that this class will be imported as a dummy object.
    pass

__all__ = [
    "monitor",
    "MonitorBase",
    "DefaultMonitor",
    "NoOpMonitor",
]

if _is_otlp_available:
    __all__.append("OTLPMonitor")


def __getattr__(item: str) -> t.Any:
    if item == "OTLPMonitor" and not _is_otlp_available:
        raise MissingDependencyException(
            "OTLPMonitor is not available, please install it with `pip install bentoml[monitor-otlp]`"
        )
    elif item in __all__:
        return importlib.import_module(f".{item}", __name__)
    else:
        raise AttributeError(f"module {__name__} has no attribute {item}")
