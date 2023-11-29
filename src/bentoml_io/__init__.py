from bentoml._internal.utils.pkg import pkg_version_info

if (ver := pkg_version_info("pydantic")) < (2,):
    raise ImportError(
        f"bentoml_io runs on pydantic>=2.0.0, but the you have {'.'.join(map(str, ver))}. "
        "Please upgrade it"
    )

# for convenience, re-export the following

from ._pydantic import add_custom_preparers
from .api import api as api
from .dependency import depends as depends
from .factory import Service as Service
from .factory import service as service

add_custom_preparers()
del add_custom_preparers
