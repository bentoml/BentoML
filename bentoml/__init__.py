# From versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from bentoml._internal.configuration import load_global_config

# TODO:
# from bentoml._internal.utils.log import configure_logging

# Inject dependencies and configurations
load_global_config()

# Configuring logging properly before loading other modules
# configure_logging()

from bentoml._internal.service import Service

__all__ = [
    "__version__",
    "Service",
]
