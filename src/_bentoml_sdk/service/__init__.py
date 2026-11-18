import typing as t

from bentoml.exceptions import BentoMLException

from .config import ServiceConfig
from .dependency import depends
from .factory import Service
from .factory import runner_service
from .factory import service

__all__ = [
    "Service",
    "ServiceConfig",
    "depends",
    "get_current_service",
    "runner_service",
    "service",
    "set_current_service",
]


_current_service: t.Any | None = None


def get_current_service() -> t.Any:
    """Return the current active service instance."""
    if _current_service is None:
        raise BentoMLException("service isn't instantiated yet")
    return _current_service


def set_current_service(service: t.Any) -> None:
    global _current_service
    _current_service = service
