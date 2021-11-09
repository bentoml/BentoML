# From versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from ._internal.configuration import load_global_config

# Inject dependencies and configurations
load_global_config()

# TODO: fix configure_logging
# from bentoml._internal.utils.log import configure_logging
# configure_logging()

from ._internal.models.store import ModelStore
from ._internal.service import Service
from ._internal.service.loader import load
from ._internal.types import Tag

# bento APIs are top-level
from .bentos import build, delete, export_bento, get, import_bento, list

# TODO: change to Store based API
models = ModelStore()

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
