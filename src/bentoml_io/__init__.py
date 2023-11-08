from bentoml._internal.utils.pkg import pkg_version_info

if (ver := pkg_version_info("pydantic")) < (2,):
    raise ImportError(
        f"bentoml_io runs on pydantic>=2.0.0, but the you have {'.'.join(map(str, ver))}. "
        "Please upgrade it"
    )

from ._pydantic import add_custom_preparers
from .api import api as api
from .servable import Servable as Servable
from .server import APIService as APIService
from .server import Service as Service

add_custom_preparers()
del add_custom_preparers
