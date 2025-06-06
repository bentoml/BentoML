from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

MODULE_ATTRS = {
    "Resource": "._internal.resource:Resource",
    "Runnable": "._internal.runner:Runnable",
    "Runner": "._internal.runner:Runner",
    "Strategy": "._internal.runner.strategy:Strategy",
    "Service": "._internal.service:Service",
    "HTTPServer": ".server:HTTPServer",
    "GrpcServer": ".server:GrpcServer",
}

__all__ = [
    "Resource",
    "Runnable",
    "Runner",
    "Strategy",
    "Service",
    "HTTPServer",
    "GrpcServer",
]

if TYPE_CHECKING:
    from ._internal.resource import Resource
    from ._internal.runner import Runnable
    from ._internal.runner import Runner
    from ._internal.runner.strategy import Strategy
    from ._internal.service import Service
    from .server import GrpcServer
    from .server import HTTPServer

else:

    def __getattr__(name: str) -> Any:
        if name in MODULE_ATTRS:
            from importlib import import_module

            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        raise AttributeError(f"module {__name__} has no attribute {name}")
