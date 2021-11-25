

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
from .configuration_retribert import (
    RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    RetriBertConfig,
)
from .tokenization_retribert import RetriBertTokenizer

_import_structure = ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
