# flake8: noqa: E402
from typing import TYPE_CHECKING

from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_global_config

# Inject dependencies and configurations
load_global_config()

from ._internal.log import configure_logging  # noqa: E402

configure_logging()

from . import models
from ._internal.service import Service
from ._internal.service.loader import load
from ._internal.types import Tag
from ._internal.utils import LazyLoader as _LazyLoader

# bento APIs are top-level
from .bentos import list  # pylint: disable=W0622
from .bentos import build, delete, export_bento, get, import_bento

if TYPE_CHECKING:
    from bentoml import keras, tensorflow
else:
    keras = _LazyLoader("keras", globals(), "bentoml.keras")
    tensorflow = _LazyLoader("tensorflow", globals(), "bentoml.tensorflow")

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
    # frameworks
    "keras",
    "tensorflow",
]
