

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
)

_import_structure = ...
if is_sentencepiece_available():
    ...
if is_tokenizers_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
