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

from .api import api, on_shutdown
from .service import depends
from .service import Service
from .service import service
from .service import runner_service
from .io_models import IODescriptor

__all__ = [
    "api",
    "on_shutdown",
    "depends",
    "Service",
    "service",
    "runner_service",
    # io descriptors
    "IODescriptor",
]
