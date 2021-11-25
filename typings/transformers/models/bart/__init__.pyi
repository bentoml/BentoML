

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)
from .configuration_bart import (
    BART_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BartConfig,
    BartOnnxConfig,
)
from .tokenization_bart import BartTokenizer

_import_structure = ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_flax_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
