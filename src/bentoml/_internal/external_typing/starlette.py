"""Type definitions for ASGI applications and middleware."""

import typing as t
from typing import Any
from typing import Dict
from typing import Protocol

# Base type definitions
ASGIApp = t.Callable[..., Any]
ASGIMessage = Dict[str, Any]
ASGIReceive = t.Callable[[], Any]
ASGIScope = Dict[str, Any]
ASGISend = t.Callable[[Dict[str, Any]], Any]


# Protocol for ASGI middleware
class AsgiMiddleware(Protocol):
    """Protocol for ASGI middleware."""

    def __call__(self, app: ASGIApp, **kwargs: Any) -> ASGIApp: ...


__all__ = [
    "AsgiMiddleware",
    "ASGIApp",
    "ASGIScope",
    "ASGISend",
    "ASGIReceive",
    "ASGIMessage",
]
