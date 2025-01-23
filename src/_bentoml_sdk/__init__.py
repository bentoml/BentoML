from __future__ import annotations
from bentoml._internal.utils.pkg import pkg_version_info
from typing_extensions import deprecated
from typing import TypeVar, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from bentoml._internal.external_typing import ASGIApp

if (ver := pkg_version_info("pydantic")) < (2,):
    raise ImportError(
        f"The new SDK runs on pydantic>=2.0.0, but the you have {'.'.join(map(str, ver))}. "
        "Please upgrade it."
    )

# ruff: noqa

from .decorators import api, on_shutdown, on_startup, asgi_app, on_deployment, task
from .service import get_current_service
from .service import depends
from .service import Service, ServiceConfig
from .service import service
from .service import runner_service
from .io_models import IODescriptor

T = TypeVar("T")


@deprecated("Deprecated in favor of `bentoml.asgi_app`")
def mount_asgi_app(
    app: ASGIApp, *, path: str = "/", name: str | None = None
) -> Callable[[T], T]:
    return asgi_app(app, path=path, name=name)


__all__ = [
    "api",
    "task",
    "on_shutdown",
    "on_startup",
    "on_deployment",
    "asgi_app",
    "mount_asgi_app",
    "depends",
    "Service",
    "ServiceConfig",
    "service",
    "runner_service",
    # io descriptors
    "IODescriptor",
    "get_current_service",
]
