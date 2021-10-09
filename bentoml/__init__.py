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

from ._internal.bento.build import build_bento as build
from ._internal.models.store import ModelStore
from ._internal.service import Service, load

# Global default ModelStore instance for direct user access
models = ModelStore()

__all__ = [
    "__version__",
    "Service",
    "load",
    "build",
    "models",
]
