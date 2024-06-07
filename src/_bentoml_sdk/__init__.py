from bentoml._internal.utils.pkg import pkg_version_info

if (ver := pkg_version_info("pydantic")) < (2,):
    raise ImportError(
        f"The new SDK runs on pydantic>=2.0.0, but the you have {'.'.join(map(str, ver))}. "
        "Please upgrade it."
    )

from ._pydantic import add_custom_preparers

add_custom_preparers()
del add_custom_preparers
# ruff: noqa

from .decorators import api, on_shutdown, mount_asgi_app, on_deployment
from .service import get_current_service
from .service import depends
from .service import Service, ServiceConfig
from .service import service
from .service import runner_service
from .io_models import IODescriptor

__all__ = [
    "api",
    "on_shutdown",
    "on_deployment",
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
