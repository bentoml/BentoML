

from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)
from ...utils.dummy_sentencepiece_objects import T5Tokenizer
from ...utils.dummy_tokenizers_objects import T5TokenizerFast
from ..t5.tokenization_t5 import T5Tokenizer
from ..t5.tokenization_t5_fast import T5TokenizerFast
from .configuration_mt5 import MT5Config

if is_sentencepiece_available():
    ...
else:
    ...
MT5Tokenizer = T5Tokenizer
if is_tokenizers_available():
    ...
else:
    ...
MT5TokenizerFast = T5TokenizerFast
_import_structure = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
