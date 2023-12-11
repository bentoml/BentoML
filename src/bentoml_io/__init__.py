from bentoml._internal.utils.pkg import pkg_version_info

if (ver := pkg_version_info("pydantic")) < (2,):
    raise ImportError(
        f"bentoml_io runs on pydantic>=2.0.0, but the you have {'.'.join(map(str, ver))}. "
        "Please upgrade it."
    )

from ._pydantic import add_custom_preparers

add_custom_preparers()
del add_custom_preparers
# ruff: noqa
# Re-export models for compatibility
from bentoml import models as models

from .api import api as api
from .dependency import depends as depends
from .factory import Service as Service
from .factory import service as service
from .factory import runner_service as runner_service
from .io_models import IODescriptor as IODescriptor
