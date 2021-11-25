

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
from .configuration_canine import CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP, CanineConfig
from .tokenization_canine import CanineTokenizer

_import_structure = ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
