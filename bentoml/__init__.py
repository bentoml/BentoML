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

# load = load_from_dir

cmd = [
    "list",
    "get",
    "delete",
    "push",
    "pull",
    "containerize",
    "serve"
]

sdk = [
    "Service",
    "env",
    "bundle",
    "load",
    "api",
    "batch_api",
]

__all__ = ["__version__", "YataiClient"] + sdk + cmd
