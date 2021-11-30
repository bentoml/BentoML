# flake8: noqa: E402
from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_global_config

# Inject dependencies and configurations
load_global_config()

from ._internal.log import configure_logging  # noqa: E402

configure_logging()

from . import models

# bento APIs are top-level
from .bentos import list  # pylint: disable=W0622
from .bentos import get, build, delete, export_bento, import_bento
from ._internal.types import Tag
from ._internal.service import Service
from ._internal.service.loader import load

__all__ = [
    "__version__",
    "Service",
    "models",
    "Tag",
    # bento APIs
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "build",
    "load",
]
