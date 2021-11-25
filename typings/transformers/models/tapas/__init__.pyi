

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_torch_available
from .configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
from .tokenization_tapas import TapasTokenizer

_import_structure = ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
