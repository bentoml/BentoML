

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
from .configuration_squeezebert import (
    SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    SqueezeBertConfig,
)
from .tokenization_squeezebert import SqueezeBertTokenizer

_import_structure = ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
