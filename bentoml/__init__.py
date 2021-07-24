# From versioneer
from ._version import get_versions

__version__ = get_versions()["version"]  # type: ignore
del get_versions

# from bentoml._internal.configuration import inject_dependencies
# from bentoml._internal.utils.log import configure_logging
#
# # Inject dependencies and configurations
# inject_dependencies()
#
# # Configuring logging properly before loading other modules
# configure_logging()

from ._internal.artifacts import ModelArtifact as BaseModel
from ._internal.artifacts import PickleArtifact
from ._internal.bundle import containerize, load
from ._internal.environment import env
from ._internal.inference_api import api, batch_api
from ._internal.repository import delete, get, ls, pull, push
from ._internal.server import serve
from ._internal.service import Service
from ._internal.yatai_client import YataiClient

__all__ = [
    "__version__",
    "YataiClient",
    "PickleArtifact",
    "BaseModel",
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
