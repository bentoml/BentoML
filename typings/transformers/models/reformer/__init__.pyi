

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)
from .configuration_reformer import (
    REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ReformerConfig,
)

_import_structure = ...
if is_sentencepiece_available():
    ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
