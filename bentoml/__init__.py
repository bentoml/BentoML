from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_global_config  # noqa: E402

# Inject dependencies and configurations
load_global_config()

from ._internal.log import configure_logging

configure_logging()

from ._internal.models.store import ModelStore  # noqa: E402
from ._internal.service import Service  # noqa: E402
from ._internal.service.loader import load  # noqa: E402
from ._internal.types import Tag  # noqa: E402
# bento APIs are top-level
from .bentos import list  # pylint: disable=W0622  # noqa: E402
from .bentos import (build, delete, export_bento, get,  # noqa: E402
                     import_bento)

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
