

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_torch_available
from .configuration_deberta_v2 import (
    DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DebertaV2Config,
)
from .tokenization_deberta_v2 import DebertaV2Tokenizer

_import_structure = ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
