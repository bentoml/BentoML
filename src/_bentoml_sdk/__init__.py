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

from .api import api as api
from .service import depends as depends
from .service import Service as Service
from .service import service as service
from .service import runner_service as runner_service
from .io_models import IODescriptor as IODescriptor
