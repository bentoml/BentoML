# From versioneer
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from bentoml._internal.configuration import inject_dependencies

# from bentoml._internal.utils.log import configure_logging

# Inject dependencies and configurations
inject_dependencies()

# Configuring logging properly before loading other modules
# configure_logging()

from bentoml._internal.bundle import containerize, load
from bentoml._internal.environment import env
from bentoml._internal.inference_api import api, batch_api
from bentoml._internal.models import Model, PickleModel
from bentoml._internal.repository import delete, get, ls, pull, push
from bentoml._internal.server import serve
from bentoml._internal.service import Service
from bentoml._internal.yatai_client import YataiClient

__all__ = [
    "__version__",
    "YataiClient",
    "PickleModel",
    "Model",
    "ls",
    "get",
    "delete",
    "push",
    "pull",
    "containerize",
    "serve",
    "Service",
    "env",
    "load",
    "api",
    "batch_api",
]
