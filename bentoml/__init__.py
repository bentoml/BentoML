# From versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

# from bentoml._internal.configuration import inject_dependencies
# from bentoml._internal.utils.log import configure_logging
#
# # Inject dependencies and configurations
# inject_dependencies()
#
# # Configuring logging properly before loading other modules
# configure_logging()

# from bentoml._internal.bundle import load_from_dir, save_to_dir  # noqa: E402
# from bentoml._internal.service import BentoService
# from bentoml._internal.service import api_decorator as api  # noqa: E402
# from bentoml._internal.service import artifacts_decorator as artifacts
# from bentoml._internal.service import env_decorator as env
# from bentoml._internal.service import save
# from bentoml._internal.service import ver_decorator as ver
# from bentoml._internal.service import web_static_content_decorator as web_static_content

from bentoml._internal import artifacts
from bentoml._internal.bundle import containerize, load
from bentoml._internal.env import env
from bentoml._internal.inference_api import api, batch_api

# load = load_from_dir
from bentoml._internal.repository import delete, get, list, pull, push
from bentoml._internal.server import serve
from bentoml._internal.service import Service
from bentoml._internal.yatai_client import YataiClient

__all__ = [
    "__version__",
    "YataiClient",
    "list",
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
